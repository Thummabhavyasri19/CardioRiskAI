import os
from io import BytesIO
from datetime import datetime

from flask import (
    Flask,
    request,
    jsonify,
    session,
    render_template,
    redirect,
    url_for,
    send_file
)

from dotenv import load_dotenv

import pandas as pd
import joblib

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from ml.risk_assessor import assess_risk
from ml.recommendations import generate_recommendations
from ml.basic_predictor import predict_basic_screening
from ml.basic_recommendations import generate_basic_recommendations

from database.db_connection import get_db_connection
from database.assessment_db import save_clinical_assessment
from database.basic_assessment_db import save_basic_assessment
from database.history_db import get_user_assessment_history


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# FLASK SECRET KEY
# ============================================================

secret_key = os.getenv("FLASK_SECRET_KEY")

if not secret_key:
    raise RuntimeError(
        "FLASK_SECRET_KEY is missing from .env"
    )

app.config["SECRET_KEY"] = secret_key


# ============================================================
# LOAD CLINICAL MACHINE LEARNING MODEL
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "random_forest.pkl"
)

model = joblib.load(MODEL_PATH)


# ============================================================
# CLINICAL SCREENING REQUIRED FIELDS
# ============================================================

REQUIRED_FIELDS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# ============================================================
# COMMON DISCLAIMER
# ============================================================

DISCLAIMER = (
    "This is a preliminary screening result based on the "
    "information provided. It is not a medical diagnosis. "
    "For medical concerns, consult a qualified healthcare "
    "professional."
)


# ============================================================
# HELPER - LOGIN CHECK
# ============================================================

def is_logged_in():
    return "user_id" in session


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# LOGIN PAGE
# ============================================================

@app.route("/login")
def login_page():

    if "user_id" in session:
        return redirect(url_for("dashboard_page"))

    return render_template(
        "login.html"
    )


# ============================================================
# REGISTER PAGE
# ============================================================

@app.route("/register")
def register_page():

    if "user_id" in session:
        return redirect(url_for("dashboard_page"))

    return render_template(
        "register.html"
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.route("/dashboard")
def dashboard_page():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    return render_template(
        "dashboard.html",
        user_name=session.get("name", "User"),
        user_email=session.get("email", "")
    )


# ============================================================
# BASIC SCREENING PAGE
# ============================================================

@app.route("/basic-screening")
def basic_screening_page():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    return render_template(
        "basic_screening.html",
        user_name=session.get("name", "User")
    )


# ============================================================
# CLINICAL SCREENING PAGE
# ============================================================

@app.route("/clinical-screening")
def clinical_screening_page():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    return render_template(
        "clinical_screening.html",
        user_name=session.get("name", "User")
    )


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history_page():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    return render_template(
        "history.html",
        user_name=session.get("name", "User")
    )


# ============================================================
# MODEL STATUS
# ============================================================

@app.route("/api/model-status")
def model_status():

    return jsonify({
        "status": "success",
        "message": "CardioRiskAI model is ready"
    })


# ============================================================
# USER REGISTRATION API
# ============================================================

@app.route("/api/register", methods=["POST"])
def register():

    db = None
    cursor = None

    try:

        data = request.get_json()

        if data is None:

            return jsonify({
                "status": "error",
                "message": "Request body must contain JSON data"
            }), 400

        name = str(
            data.get("name", "")
        ).strip()

        email = str(
            data.get("email", "")
        ).strip().lower()

        password = data.get(
            "password",
            ""
        )

        # ----------------------------------------------------
        # Validate name
        # ----------------------------------------------------

        if not name:

            return jsonify({
                "status": "error",
                "message": "Name is required"
            }), 400

        # ----------------------------------------------------
        # Validate email
        # ----------------------------------------------------

        if not email:

            return jsonify({
                "status": "error",
                "message": "Email is required"
            }), 400

        if "@" not in email or "." not in email:

            return jsonify({
                "status": "error",
                "message": "Enter a valid email address"
            }), 400

        # ----------------------------------------------------
        # Validate password
        # ----------------------------------------------------

        if not password:

            return jsonify({
                "status": "error",
                "message": "Password is required"
            }), 400

        if len(password) < 8:

            return jsonify({
                "status": "error",
                "message": (
                    "Password must contain at least 8 characters"
                )
            }), 400

        # ----------------------------------------------------
        # Hash password
        # ----------------------------------------------------

        password_hash = generate_password_hash(
            password
        )

        # ----------------------------------------------------
        # Database
        # ----------------------------------------------------

        db = get_db_connection()

        cursor = db.cursor(
            dictionary=True
        )

        # ----------------------------------------------------
        # Check existing email
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT user_id
            FROM users
            WHERE email = %s
            """,
            (email,)
        )

        existing_user = cursor.fetchone()

        if existing_user:

            return jsonify({
                "status": "error",
                "message": (
                    "An account with this email already exists"
                )
            }), 409

        # ----------------------------------------------------
        # Insert user
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                password_hash
            )
            VALUES
            (
                %s,
                %s,
                %s
            )
            """,
            (
                name,
                email,
                password_hash
            )
        )

        db.commit()

        user_id = cursor.lastrowid

        return jsonify({
            "status": "success",
            "message": "User registered successfully",
            "user_id": user_id
        }), 201

    except Exception as e:

        if db:
            db.rollback()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

    finally:

        if cursor:
            cursor.close()

        if db:
            db.close()


# ============================================================
# USER LOGIN API
# ============================================================

@app.route("/api/login", methods=["POST"])
def login():

    db = None
    cursor = None

    try:

        data = request.get_json()

        if data is None:

            return jsonify({
                "status": "error",
                "message": "Request body must contain JSON data"
            }), 400

        email = str(
            data.get("email", "")
        ).strip().lower()

        password = data.get(
            "password",
            ""
        )

        if not email:

            return jsonify({
                "status": "error",
                "message": "Email is required"
            }), 400

        if not password:

            return jsonify({
                "status": "error",
                "message": "Password is required"
            }), 400

        # ----------------------------------------------------
        # Database
        # ----------------------------------------------------

        db = get_db_connection()

        cursor = db.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT
                user_id,
                name,
                email,
                password_hash
            FROM users
            WHERE email = %s
            """,
            (email,)
        )

        user = cursor.fetchone()

        if user is None:

            return jsonify({
                "status": "error",
                "message": "Invalid email or password"
            }), 401

        # ----------------------------------------------------
        # Check password
        # ----------------------------------------------------

        password_correct = check_password_hash(
            user["password_hash"],
            password
        )

        if not password_correct:

            return jsonify({
                "status": "error",
                "message": "Invalid email or password"
            }), 401

        # ----------------------------------------------------
        # Create session
        # ----------------------------------------------------

        session.clear()

        session["user_id"] = user["user_id"]
        session["name"] = user["name"]
        session["email"] = user["email"]

        return jsonify({
            "status": "success",
            "message": "Login successful",
            "user": {
                "user_id": user["user_id"],
                "name": user["name"],
                "email": user["email"]
            }
        }), 200

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

    finally:

        if cursor:
            cursor.close()

        if db:
            db.close()


# ============================================================
# CURRENT USER API
# ============================================================

@app.route("/api/me")
def current_user():

    if "user_id" not in session:

        return jsonify({
            "status": "error",
            "message": "User is not logged in"
        }), 401

    return jsonify({
        "status": "success",
        "user": {
            "user_id": session["user_id"],
            "name": session.get("name", "User"),
            "email": session.get("email", "")
        }
    })


# ============================================================
# LOGOUT API
# ============================================================

@app.route("/api/logout", methods=["POST"])
def logout():

    session.clear()

    return jsonify({
        "status": "success",
        "message": "Logout successful"
    })


# ============================================================
# CLINICAL SCREENING API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Check login
        # ----------------------------------------------------

        if "user_id" not in session:

            return jsonify({
                "status": "error",
                "message": (
                    "Please login before starting a screening"
                )
            }), 401

        # ----------------------------------------------------
        # Get JSON
        # ----------------------------------------------------

        data = request.get_json()

        if data is None:

            return jsonify({
                "status": "error",
                "message": (
                    "Request body must contain JSON data"
                )
            }), 400

        # ----------------------------------------------------
        # Check missing fields
        # ----------------------------------------------------

        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in data
        ]

        if missing_fields:

            return jsonify({
                "status": "error",
                "message": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # ----------------------------------------------------
        # Create patient dataframe
        # ----------------------------------------------------

        patient = pd.DataFrame([{
            "age": data["age"],
            "sex": data["sex"],
            "cp": data["cp"],
            "trestbps": data["trestbps"],
            "chol": data["chol"],
            "fbs": data["fbs"],
            "restecg": data["restecg"],
            "thalach": data["thalach"],
            "exang": data["exang"],
            "oldpeak": data["oldpeak"],
            "slope": data["slope"],
            "ca": data["ca"],
            "thal": data["thal"]
        }])

        # ----------------------------------------------------
        # Convert numeric values
        # ----------------------------------------------------

        for column in REQUIRED_FIELDS:

            patient[column] = pd.to_numeric(
                patient[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Check invalid numeric values
        # ----------------------------------------------------

        if patient[REQUIRED_FIELDS].isnull().any().any():

            return jsonify({
                "status": "error",
                "message": (
                    "All clinical screening fields must "
                    "contain valid numeric values"
                )
            }), 400

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        age = patient.loc[0, "age"]
        sex = patient.loc[0, "sex"]
        cp = patient.loc[0, "cp"]
        trestbps = patient.loc[0, "trestbps"]
        chol = patient.loc[0, "chol"]
        fbs = patient.loc[0, "fbs"]
        restecg = patient.loc[0, "restecg"]
        thalach = patient.loc[0, "thalach"]
        exang = patient.loc[0, "exang"]
        oldpeak = patient.loc[0, "oldpeak"]
        slope = patient.loc[0, "slope"]
        ca = patient.loc[0, "ca"]
        thal = patient.loc[0, "thal"]

        if age < 18 or age > 120:

            return jsonify({
                "status": "error",
                "message": "Age must be between 18 and 120"
            }), 400

        if sex not in [0, 1]:

            return jsonify({
                "status": "error",
                "message": "Sex must be 0 or 1"
            }), 400

        if cp not in [1, 2, 3, 4]:

            return jsonify({
                "status": "error",
                "message": (
                    "Chest pain type must be between 1 and 4"
                )
            }), 400

        if trestbps <= 0:

            return jsonify({
                "status": "error",
                "message": (
                    "Resting blood pressure must be greater than 0"
                )
            }), 400

        if chol <= 0:

            return jsonify({
                "status": "error",
                "message": (
                    "Cholesterol must be greater than 0"
                )
            }), 400

        if fbs not in [0, 1]:

            return jsonify({
                "status": "error",
                "message": "Fasting blood sugar must be 0 or 1"
            }), 400

        if restecg not in [0, 1, 2]:

            return jsonify({
                "status": "error",
                "message": (
                    "Resting ECG must be 0, 1 or 2"
                )
            }), 400

        if thalach <= 0:

            return jsonify({
                "status": "error",
                "message": (
                    "Maximum heart rate must be greater than 0"
                )
            }), 400

        if exang not in [0, 1]:

            return jsonify({
                "status": "error",
                "message": (
                    "Exercise-induced angina must be 0 or 1"
                )
            }), 400

        if oldpeak < 0:

            return jsonify({
                "status": "error",
                "message": "Oldpeak cannot be negative"
            }), 400

        if slope not in [1, 2, 3]:

            return jsonify({
                "status": "error",
                "message": (
                    "ST segment slope must be between 1 and 3"
                )
            }), 400

        if ca not in [0, 1, 2, 3]:

            return jsonify({
                "status": "error",
                "message": (
                    "Major vessels value must be between 0 and 3"
                )
            }), 400

        if thal not in [3, 6, 7]:

            return jsonify({
                "status": "error",
                "message": (
                    "Thal value must be 3, 6 or 7"
                )
            }), 400

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        prediction = model.predict(
            patient
        )[0]

        probability = model.predict_proba(
            patient
        )[0][1]

        # ----------------------------------------------------
        # Risk assessment
        # ----------------------------------------------------

        risk_result = assess_risk(
            probability
        )

        risk_score = float(
            risk_result["risk_score"]
        )

        risk_level = risk_result[
            "risk_level"
        ]

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        patient_data = patient.iloc[
            0
        ].to_dict()

        recommendations = generate_recommendations(
            patient_data
        )

        # ----------------------------------------------------
        # Save assessment
        # ----------------------------------------------------

        assessment_id = save_clinical_assessment(
            user_id=session["user_id"],
            patient=patient_data,
            risk_score=risk_score,
            risk_level=risk_level,
            prediction=int(prediction)
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "screening_type": "Clinical Screening",
            "recommendations": recommendations,
            "assessment_id": assessment_id,
            "disclaimer": DISCLAIMER,
            "message": DISCLAIMER
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ============================================================
# BASIC SCREENING API
# ============================================================

@app.route("/api/basic-predict", methods=["POST"])
def basic_predict():

    try:

        # ----------------------------------------------------
        # Check login
        # ----------------------------------------------------

        if "user_id" not in session:

            return jsonify({
                "status": "error",
                "message": (
                    "Please login before starting a screening"
                )
            }), 401

        # ----------------------------------------------------
        # Get JSON
        # ----------------------------------------------------

        data = request.get_json()

        if data is None:

            return jsonify({
                "status": "error",
                "message": (
                    "Request body must contain JSON data"
                )
            }), 400

        # ----------------------------------------------------
        # Required basic fields
        # ----------------------------------------------------

        basic_fields = [
            "age",
            "height_cm",
            "weight_kg",
            "age_group",
            "sex",
            "bmi",
            "smoking",
            "physical_inactivity",
            "high_blood_pressure",
            "diabetes_status",
            "general_health",
            "sleep_hours",
            "alcohol_use",
            "previous_stroke"
        ]

        missing_fields = [
            field
            for field in basic_fields
            if field not in data
        ]

        if missing_fields:

            return jsonify({
                "status": "error",
                "message": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # ----------------------------------------------------
        # Convert values
        # ----------------------------------------------------

        try:

            age = int(data["age"])
            height_cm = float(data["height_cm"])
            weight_kg = float(data["weight_kg"])

            sex = int(data["sex"])
            smoking = int(data["smoking"])

            physical_inactivity = int(
                data["physical_inactivity"]
            )

            high_blood_pressure = int(
                data["high_blood_pressure"]
            )

            diabetes_status = int(
                data["diabetes_status"]
            )

            general_health = int(
                data["general_health"]
            )

            sleep_hours = int(
                data["sleep_hours"]
            )

            alcohol_use = int(
                data["alcohol_use"]
            )

            previous_stroke = int(
                data["previous_stroke"]
            )

        except (TypeError, ValueError):

            return jsonify({
                "status": "error",
                "message": (
                    "Basic screening fields contain "
                    "invalid values"
                )
            }), 400

        # ----------------------------------------------------
        # Calculate BMI from height and weight
        # ----------------------------------------------------

        height_m = height_cm / 100

        if height_m <= 0:

            return jsonify({
                "status": "error",
                "message": "Height must be greater than 0"
            }), 400

        calculated_bmi = (
            weight_kg /
            (height_m * height_m)
        )

        bmi = round(
            calculated_bmi,
            2
        )

        # ----------------------------------------------------
        # Calculate age group
        # ----------------------------------------------------

        if 18 <= age <= 24:
            age_group = 1

        elif 25 <= age <= 29:
            age_group = 2

        elif 30 <= age <= 34:
            age_group = 3

        elif 35 <= age <= 39:
            age_group = 4

        elif 40 <= age <= 44:
            age_group = 5

        elif 45 <= age <= 49:
            age_group = 6

        elif 50 <= age <= 54:
            age_group = 7

        elif 55 <= age <= 59:
            age_group = 8

        elif 60 <= age <= 64:
            age_group = 9

        elif 65 <= age <= 69:
            age_group = 10

        elif 70 <= age <= 74:
            age_group = 11

        elif 75 <= age <= 79:
            age_group = 12

        else:
            age_group = 13

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if age < 18 or age > 120:

            return jsonify({
                "status": "error",
                "message": "Age must be between 18 and 120"
            }), 400

        if height_cm < 100 or height_cm > 250:

            return jsonify({
                "status": "error",
                "message": (
                    "Height must be between 100 and 250 cm"
                )
            }), 400

        if weight_kg < 20 or weight_kg > 300:

            return jsonify({
                "status": "error",
                "message": (
                    "Weight must be between 20 and 300 kg"
                )
            }), 400

        if sex not in [0, 1]:

            return jsonify({
                "status": "error",
                "message": "Sex must be 0 or 1"
            }), 400

        if bmi < 10 or bmi > 100:

            return jsonify({
                "status": "error",
                "message": (
                    "Calculated BMI must be between 10 and 100"
                )
            }), 400

        if smoking not in [0, 1]:

            return jsonify({
                "status": "error",
                "message": "Smoking must be 0 or 1"
            }), 400

        if physical_inactivity not in [0, 1]:

            return jsonify({
                "status": "error",
                "message": (
                    "Physical inactivity must be 0 or 1"
                )
            }), 400

        if high_blood_pressure not in [0, 1]:

            return jsonify({
                "status": "error",
                "message": (
                    "High blood pressure must be 0 or 1"
                )
            }), 400

        if diabetes_status not in [1, 2, 3, 4]:

            return jsonify({
                "status": "error",
                "message": (
                    "Diabetes status must be between 1 and 4"
                )
            }), 400

        if general_health not in [1, 2, 3, 4, 5]:

            return jsonify({
                "status": "error",
                "message": (
                    "General health must be between 1 and 5"
                )
            }), 400

        if sleep_hours < 1 or sleep_hours > 24:

            return jsonify({
                "status": "error",
                "message": (
                    "Sleep hours must be between 1 and 24"
                )
            }), 400

        if alcohol_use not in [0, 1]:

            return jsonify({
                "status": "error",
                "message": "Alcohol use must be 0 or 1"
            }), 400

        if previous_stroke not in [0, 1]:

            return jsonify({
                "status": "error",
                "message": "Previous stroke must be 0 or 1"
            }), 400

        # ----------------------------------------------------
        # Create model input
        # ----------------------------------------------------

        patient = {
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "age_group": age_group,
            "sex": sex,
            "bmi": bmi,
            "smoking": smoking,
            "physical_inactivity": physical_inactivity,
            "high_blood_pressure": high_blood_pressure,
            "diabetes_status": diabetes_status,
            "general_health": general_health,
            "sleep_hours": sleep_hours,
            "alcohol_use": alcohol_use,
            "previous_stroke": previous_stroke
        }

        # ----------------------------------------------------
        # Basic ML prediction
        # ----------------------------------------------------

        result = predict_basic_screening(
            patient
        )

        screening_score = float(
            result["screening_score"]
        )

        prediction = int(
            result["prediction"]
        )

        screening_result = result[
            "screening_result"
        ]

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        recommendations = generate_basic_recommendations(
            patient
        )

        # ----------------------------------------------------
        # Save assessment
        # ----------------------------------------------------

        assessment_id = save_basic_assessment(
            user_id=session["user_id"],
            patient=patient,
            screening_score=screening_score,
            screening_result=screening_result,
            prediction=prediction
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({
            "status": "success",
            "screening_type": "Basic Screening",
            "screening_score": screening_score,
            "screening_result": screening_result,
            "assessment_id": assessment_id,
            "recommendations": recommendations,
            "disclaimer": DISCLAIMER,
            "message": DISCLAIMER
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ============================================================
# HISTORY API
# ============================================================

@app.route("/api/history")
def history_api():

    try:

        if "user_id" not in session:

            return jsonify({
                "status": "error",
                "message": (
                    "Please login to view your screening history"
                )
            }), 401

        user_id = session["user_id"]

        history_data = get_user_assessment_history(
            user_id
        )

        for assessment in history_data:

            if assessment.get("risk_score") is not None:

                assessment["risk_score"] = float(
                    assessment["risk_score"]
                )

            if assessment.get("created_at"):

                assessment["created_at"] = (
                    assessment["created_at"].isoformat()
                )

            assessment["print_url"] = url_for(
                "print_report",
                assessment_id=assessment["assessment_id"]
            )

            assessment["report_url"] = url_for(
                "download_report",
                assessment_id=assessment["assessment_id"]
            )

        return jsonify({
            "status": "success",
            "history": history_data
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ============================================================
# GET ASSESSMENT REPORT DATA
# ============================================================

def get_report_data(assessment_id):

    if "user_id" not in session:
        return None

    db = get_db_connection()

    cursor = db.cursor(
        dictionary=True
    )

    try:

        # ----------------------------------------------------
        # Get assessment
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                assessment_id,
                user_id,
                assessment_type,
                risk_score,
                risk_level,
                prediction,
                created_at
            FROM assessments
            WHERE assessment_id = %s
            AND user_id = %s
            """,
            (
                assessment_id,
                session["user_id"]
            )
        )

        assessment = cursor.fetchone()

        if assessment is None:
            return None

        # ----------------------------------------------------
        # Get user
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                user_id,
                name,
                email
            FROM users
            WHERE user_id = %s
            """,
            (
                session["user_id"],
            )
        )

        user = cursor.fetchone()

        # ----------------------------------------------------
        # Clinical inputs
        # ----------------------------------------------------

        if assessment["assessment_type"] == "Clinical Screening":

            cursor.execute(
                """
                SELECT
                    age,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal
                FROM clinical_inputs
                WHERE assessment_id = %s
                """,
                (assessment_id,)
            )

            inputs = cursor.fetchone()

        # ----------------------------------------------------
        # Basic inputs
        # ----------------------------------------------------

        elif assessment["assessment_type"] == "Basic Screening":

            cursor.execute(
                """
                SELECT
                    age,
                    height_cm,
                    weight_kg,
                    age_group,
                    sex,
                    bmi,
                    smoking,
                    physical_inactivity,
                    high_blood_pressure,
                    diabetes_status,
                    general_health,
                    sleep_hours,
                    alcohol_use,
                    previous_stroke
                FROM basic_inputs
                WHERE assessment_id = %s
                """,
                (assessment_id,)
            )

            inputs = cursor.fetchone()

        else:

            inputs = None

        if inputs is None:
            return None

        return {
            "assessment": assessment,
            "user": user,
            "inputs": inputs
        }

    finally:

        cursor.close()
        db.close()


# ============================================================
# PRINTABLE REPORT
# ============================================================

@app.route("/report/<int:assessment_id>")
def print_report(assessment_id):

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    report_data = get_report_data(
        assessment_id
    )

    if report_data is None:

        return (
            "Assessment not found.",
            404
        )

    assessment = report_data["assessment"]
    user = report_data["user"]
    inputs = report_data["inputs"]

    risk_score = float(
        assessment["risk_score"]
    )

    recommendations = []

    # --------------------------------------------------------
    # Generate recommendations again for report
    # --------------------------------------------------------

    if assessment["assessment_type"] == "Clinical Screening":

        clinical_patient = {
            key: inputs[key]
            for key in inputs
        }

        recommendations = generate_recommendations(
            clinical_patient
        )

    elif assessment["assessment_type"] == "Basic Screening":

        basic_patient = {
            key: inputs[key]
            for key in inputs
        }

        recommendations = generate_basic_recommendations(
            basic_patient
        )

    # --------------------------------------------------------
    # Prepare report
    # --------------------------------------------------------

    report = {
        "assessment": assessment,
        "user": user,
        "inputs": inputs,
        "risk_score": risk_score,
        "recommendations": recommendations,
        "disclaimer": DISCLAIMER
    }

    return render_template(
        "report.html",
        report=report,
        assessment=assessment,
        user=user,
        inputs=inputs,
        risk_score=risk_score,
        recommendations=recommendations,
        disclaimer=DISCLAIMER
    )


# ============================================================
# DOWNLOAD REPORT AS PDF
# ============================================================

@app.route("/api/report/<int:assessment_id>")
def download_report(assessment_id):

    if "user_id" not in session:

        return jsonify({
            "status": "error",
            "message": "Please login first"
        }), 401

    try:

        report_data = get_report_data(
            assessment_id
        )

        if report_data is None:

            return jsonify({
                "status": "error",
                "message": "Assessment not found"
            }), 404

        assessment = report_data["assessment"]
        user = report_data["user"]
        inputs = report_data["inputs"]

        risk_score = float(
            assessment["risk_score"]
        )

        # ----------------------------------------------------
        # Generate recommendations
        # ----------------------------------------------------

        if assessment["assessment_type"] == "Clinical Screening":

            recommendations = generate_recommendations(
                dict(inputs)
            )

        else:

            recommendations = generate_basic_recommendations(
                dict(inputs)
            )

        # ----------------------------------------------------
        # Try ReportLab
        # ----------------------------------------------------

        try:

            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.enums import TA_CENTER
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle
            )

        except ImportError:

            return jsonify({
                "status": "error",
                "message": (
                    "ReportLab is not installed. "
                    "Run: pip install reportlab"
                )
            }), 500

        # ----------------------------------------------------
        # Create PDF
        # ----------------------------------------------------

        pdf_buffer = BytesIO()

        document = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()

        title_style = styles["Title"]
        title_style.alignment = TA_CENTER

        heading_style = styles["Heading2"]
        normal_style = styles["BodyText"]

        elements = []

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        elements.append(
            Paragraph(
                "CardioRiskAI",
                title_style
            )
        )

        elements.append(
            Paragraph(
                "Heart Health Screening Report",
                heading_style
            )
        )

        elements.append(
            Spacer(
                1,
                12
            )
        )

        # ----------------------------------------------------
        # User information
        # ----------------------------------------------------

        user_table = Table([
            ["Report Type", assessment["assessment_type"]],
            ["Date", str(assessment["created_at"])],
            ["Patient", user["name"]],
            ["Email", user["email"]]
        ])

        user_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.whitesmoke
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ])
        )

        elements.append(
            user_table
        )

        elements.append(
            Spacer(
                1,
                15
            )
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        elements.append(
            Paragraph(
                "Screening Result",
                heading_style
            )
        )

        result_table = Table([
            ["Risk Score", f"{risk_score:.2f}%"],
            ["Risk Category", str(assessment["risk_level"])]
        ])

        result_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.whitesmoke
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ])
        )

        elements.append(
            result_table
        )

        elements.append(
            Spacer(
                1,
                15
            )
        )

        # ----------------------------------------------------
        # Entered information
        # ----------------------------------------------------

        elements.append(
            Paragraph(
                "Entered Information",
                heading_style
            )
        )

        input_rows = [
            ["Field", "Value"]
        ]

        if assessment["assessment_type"] == "Clinical Screening":

            clinical_labels = {
                "age": "Age",
                "sex": "Gender",
                "cp": "Chest Pain Type",
                "trestbps": "Resting Blood Pressure",
                "chol": "Cholesterol",
                "fbs": "Fasting Blood Sugar",
                "restecg": "Resting ECG",
                "thalach": "Maximum Heart Rate",
                "exang": "Exercise-induced Angina",
                "oldpeak": "Oldpeak",
                "slope": "ST Segment Slope",
                "ca": "Major Vessels",
                "thal": "Thal"
            }

            for key, label in clinical_labels.items():

                input_rows.append([
                    label,
                    str(inputs.get(key, ""))
                ])

        else:

            basic_labels = {
                "age": "Age",
                "height_cm": "Height",
                "weight_kg": "Weight",
                "bmi": "BMI",
                "sex": "Gender",
                "smoking": "Smoking",
                "physical_inactivity": "Physical Inactivity",
                "high_blood_pressure": "High Blood Pressure",
                "diabetes_status": "Diabetes Status",
                "general_health": "General Health",
                "sleep_hours": "Sleep Hours",
                "alcohol_use": "Alcohol Use",
                "previous_stroke": "Previous Stroke"
            }

            for key, label in basic_labels.items():

                input_rows.append([
                    label,
                    str(inputs.get(key, ""))
                ])

        input_table = Table(
            input_rows,
            colWidths=[220, 250]
        )

        input_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.whitesmoke
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                )
            ])
        )

        elements.append(
            input_table
        )

        elements.append(
            Spacer(
                1,
                15
            )
        )

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        elements.append(
            Paragraph(
                "Personalized Recommendations",
                heading_style
            )
        )

        if recommendations:

            for recommendation in recommendations:

                elements.append(
                    Paragraph(
                        "• " + str(recommendation),
                        normal_style
                    )
                )

                elements.append(
                    Spacer(
                        1,
                        5
                    )
                )

        else:

            elements.append(
                Paragraph(
                    "No recommendations available.",
                    normal_style
                )
            )

        elements.append(
            Spacer(
                1,
                15
            )
        )

        # ----------------------------------------------------
        # Disclaimer
        # ----------------------------------------------------

        elements.append(
            Paragraph(
                "<b>Important:</b> " + DISCLAIMER,
                normal_style
            )
        )

        # ----------------------------------------------------
        # Build PDF
        # ----------------------------------------------------

        document.build(
            elements
        )

        pdf_buffer.seek(0)

        filename = (
            f"CardioRiskAI_Report_{assessment_id}.pdf"
        )

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ============================================================
# DELETE HISTORY
# ============================================================

@app.route(
    "/api/history/<int:assessment_id>",
    methods=["DELETE"]
)
def delete_history(assessment_id):

    if "user_id" not in session:

        return jsonify({
            "status": "error",
            "message": "Please login first"
        }), 401

    db = None
    cursor = None

    try:

        db = get_db_connection()

        cursor = db.cursor(
            dictionary=True
        )

        # ----------------------------------------------------
        # Verify ownership
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                assessment_id,
                assessment_type
            FROM assessments
            WHERE assessment_id = %s
            AND user_id = %s
            """,
            (
                assessment_id,
                session["user_id"]
            )
        )

        assessment = cursor.fetchone()

        if assessment is None:

            return jsonify({
                "status": "error",
                "message": "Assessment not found"
            }), 404

        # ----------------------------------------------------
        # Delete corresponding input record
        # ----------------------------------------------------

        if assessment["assessment_type"] == "Clinical Screening":

            cursor.execute(
                """
                DELETE FROM clinical_inputs
                WHERE assessment_id = %s
                """,
                (assessment_id,)
            )

        elif assessment["assessment_type"] == "Basic Screening":

            cursor.execute(
                """
                DELETE FROM basic_inputs
                WHERE assessment_id = %s
                """,
                (assessment_id,)
            )

        # ----------------------------------------------------
        # Delete assessment
        # ----------------------------------------------------

        cursor.execute(
            """
            DELETE FROM assessments
            WHERE assessment_id = %s
            AND user_id = %s
            """,
            (
                assessment_id,
                session["user_id"]
            )
        )

        db.commit()

        return jsonify({
            "status": "success",
            "message": (
                "Screening history deleted successfully"
            )
        })

    except Exception as e:

        if db:
            db.rollback()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

    finally:

        if cursor:
            cursor.close()

        if db:
            db.close()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )