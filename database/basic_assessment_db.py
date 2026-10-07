from database.db_connection import get_db_connection


# =======================================
# Save Basic Screening assessment
# =======================================
def save_basic_assessment(
    user_id,
    patient,
    screening_score,
    screening_result,
    prediction
):

    db = get_db_connection()
    cursor = db.cursor()

    try:

        # ---------------------------------------
        # Store short database category
        # ---------------------------------------
        if prediction == 1:
            stored_risk_level = "Higher"
        else:
            stored_risk_level = "Lower"

        # ---------------------------------------
        # 1. Save assessment result
        # ---------------------------------------
        cursor.execute(
            """
            INSERT INTO assessments
            (
                user_id,
                assessment_type,
                risk_score,
                risk_level,
                prediction
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                user_id,
                "Basic Screening",
                float(screening_score),
                stored_risk_level,
                int(prediction)
            )
        )

        assessment_id = cursor.lastrowid

        # ---------------------------------------
        # 2. Save all Basic Screening inputs
        # ---------------------------------------
        cursor.execute(
            """
            INSERT INTO basic_inputs
            (
                assessment_id,
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
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                assessment_id,
                int(patient["age"]),
                float(patient["height_cm"]),
                float(patient["weight_kg"]),
                int(patient["age_group"]),
                int(patient["sex"]),
                float(patient["bmi"]),
                int(patient["smoking"]),
                int(patient["physical_inactivity"]),
                int(patient["high_blood_pressure"]),
                int(patient["diabetes_status"]),
                int(patient["general_health"]),
                int(patient["sleep_hours"]),
                int(patient["alcohol_use"]),
                int(patient["previous_stroke"])
            )
        )

        # ---------------------------------------
        # 3. Commit
        # ---------------------------------------
        db.commit()

        return assessment_id

    except Exception:

        db.rollback()
        raise

    finally:

        cursor.close()
        db.close()