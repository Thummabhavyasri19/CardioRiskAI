from database.db_connection import get_db_connection


# =========================================================
# GET COMPLETE SCREENING HISTORY FOR LOGGED-IN USER
# =========================================================

def get_user_assessment_history(user_id):

    db = get_db_connection()

    cursor = db.cursor(dictionary=True)

    try:

        # =================================================
        # GET COMMON ASSESSMENT INFORMATION
        # =================================================

        cursor.execute(
            """
            SELECT
                a.assessment_id,
                a.assessment_type,
                a.risk_score,
                a.risk_level,
                a.prediction,
                a.created_at
            FROM assessments a
            WHERE a.user_id = %s
            ORDER BY a.created_at DESC
            """,
            (user_id,)
        )

        assessments = cursor.fetchall()


        # =================================================
        # GET INPUT DETAILS FOR EACH ASSESSMENT
        # =================================================

        for assessment in assessments:

            assessment_id = assessment["assessment_id"]

            assessment_type = assessment[
                "assessment_type"
            ]


            # =================================================
            # CLINICAL SCREENING
            # =================================================

            if assessment_type == "Clinical Screening":

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

                clinical = cursor.fetchone()

                assessment["inputs"] = clinical


            # =================================================
            # BASIC SCREENING
            # =================================================

            elif assessment_type == "Basic Screening":

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

                basic = cursor.fetchone()

                assessment["inputs"] = basic


            else:

                assessment["inputs"] = None


        return assessments

    finally:

        cursor.close()

        db.close()