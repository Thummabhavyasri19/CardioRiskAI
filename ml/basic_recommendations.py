# =======================================
# Basic Screening Recommendations
# =======================================

from ml import recommendations


def generate_basic_recommendations(patient):

    recommendations = []

    # ---------------------------------------
    # Smoking
    # ---------------------------------------
    if patient["smoking"] == 1:

        recommendations.append(
            "Avoid smoking and tobacco products."
        )

    # ---------------------------------------
    # Physical inactivity
    # ---------------------------------------
    if patient["physical_inactivity"] == 1:

        recommendations.append(
            "Try to include regular physical activity that is appropriate for your health and abilities."
        )

    # ---------------------------------------
    # Blood pressure history
    # ---------------------------------------
    if patient["high_blood_pressure"] == 1:

        recommendations.append(
            "Keep monitoring your blood pressure and follow the advice given by your healthcare professional."
        )

    # ---------------------------------------
    # Diabetes / Prediabetes
    # ---------------------------------------
    if patient["diabetes_status"] in [1, 4]:

        recommendations.append(
            "Maintain regular blood-glucose follow-up and follow the health advice provided by your healthcare professional."
        )

    # ---------------------------------------
    # BMI
    # ---------------------------------------
    if patient["bmi"] < 18.5:

        recommendations.append(
            "Consider discussing healthy nutrition and weight goals with a healthcare professional."
        )

    elif patient["bmi"] >= 25:

        recommendations.append(
            "Aim for a healthy weight through balanced nutrition and appropriate physical activity."
        )

    # ---------------------------------------
    # Alcohol
    # ---------------------------------------
    if patient["alcohol_use"] == 1:

        recommendations.append(
            "Avoid harmful alcohol use and consider reducing alcohol consumption."
        )

    # ---------------------------------------
    # Sleep
    # ---------------------------------------
    if patient["sleep_hours"] < 7 or patient["sleep_hours"] > 9:

        recommendations.append(
            "Try to maintain a regular and adequate sleep schedule."
        )

    # ---------------------------------------
    # Previous stroke/history
    # ---------------------------------------
    if patient["previous_stroke"] == 1:

        recommendations.append(
            "Continue regular follow-up with your healthcare professional and keep your previous medical records available."
        )

    # =======================================
    # General heart-health advice
    # =======================================

    recommendations.append(
        "Choose a balanced diet with vegetables, fruits, whole grains and healthy protein sources."
    )

    recommendations.append(
        "Limit excess salt, added sugars and unhealthy fats."
    )

    recommendations.append(
        "Maintain regular health check-ups and monitor important health measurements when recommended."
    )
    return recommendations