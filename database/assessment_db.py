from database.db_connection import get_db_connection


# =======================================
# Save clinical assessment
# =======================================
def save_clinical_assessment(
    user_id,
    patient,
    risk_score,
    risk_level,
    prediction
):

    db = get_db_connection()

    cursor = db.cursor()

    try:

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
                "Clinical Screening",
                float(risk_score),
                risk_level,
                int(prediction)
            )
        )

        assessment_id = cursor.lastrowid

        # ---------------------------------------
        # 2. Save clinical inputs
        # ---------------------------------------
        cursor.execute(
            """
            INSERT INTO clinical_inputs
            (
                assessment_id,
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
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                assessment_id,
                int(patient["age"]),
                int(patient["sex"]),
                int(patient["cp"]),
                float(patient["trestbps"]),
                float(patient["chol"]),
                int(patient["fbs"]),
                int(patient["restecg"]),
                float(patient["thalach"]),
                int(patient["exang"]),
                float(patient["oldpeak"]),
                int(patient["slope"]),
                float(patient["ca"]),
                int(patient["thal"])
            )
        )

        # ---------------------------------------
        # 3. Commit both records
        # ---------------------------------------
        db.commit()

        return assessment_id

    except Exception:

        # ---------------------------------------
        # Undo database changes if something fails
        # ---------------------------------------
        db.rollback()

        raise

    finally:

        cursor.close()
        db.close()