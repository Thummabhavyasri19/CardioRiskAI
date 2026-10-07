import pandas as pd
import joblib


# =======================================
# Load tuned Basic Screening model
# =======================================

model_info = joblib.load(
    "models/basic_logistic_tuned.pkl"
)

model = model_info["model"]
threshold = model_info["threshold"]


# =======================================
# Convert user-friendly input
# into model input
# =======================================

def prepare_basic_input(data):
    """
    Convert user-friendly Basic Screening
    values into the exact format expected
    by the trained model.
    """

    patient = pd.DataFrame([{

        "age_group": data["age_group"],

        "sex": data["sex"],

        "bmi": data["bmi"],

        "smoking": data["smoking"],

        "physical_inactivity": data["physical_inactivity"],

        "high_blood_pressure": data["high_blood_pressure"],

        "diabetes_status": data["diabetes_status"],

        "general_health": data["general_health"],

        "sleep_hours": data["sleep_hours"],

        "alcohol_use": data["alcohol_use"],

        "previous_stroke": data["previous_stroke"]

    }])

    return patient


# =======================================
# Make Basic Screening prediction
# =======================================

def predict_basic_screening(data):

    patient = prepare_basic_input(data)

    # ---------------------------------------
    # Model probability
    # ---------------------------------------

    probability = model.predict_proba(
        patient
    )[0][1]

    screening_score = round(
        probability * 100,
        2
    )

    # ---------------------------------------
    # Apply tuned classification threshold
    # ---------------------------------------

    prediction = int(
        probability >= threshold
    )

    # ---------------------------------------
    # Screening result
    # ---------------------------------------

    if prediction == 1:

        screening_result = "Higher screening signal"

    else:

        screening_result = "Lower screening signal"

    return {
        "screening_score": screening_score,
        "prediction": prediction,
        "screening_result": screening_result
    }