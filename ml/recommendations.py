# ---------------------------------------
# Personalized Recommendation Module
# ---------------------------------------

def generate_recommendations(patient):
    """
    Generate general heart-health recommendations
    from the clinical information entered by the user.

    These are educational recommendations and are
    not a medical diagnosis or treatment plan.
    """

    recommendations = []

    # ---------------------------------------
    # Blood pressure
    # ---------------------------------------
    if patient["trestbps"] >= 130:
        recommendations.append(
            "Monitor your blood pressure regularly and discuss elevated readings with a healthcare professional."
        )

    # ---------------------------------------
    # Cholesterol
    # ---------------------------------------
    if patient["chol"] >= 200:
        recommendations.append(
            "Choose more vegetables, fruits, whole grains and healthy protein sources, and limit foods high in saturated fat."
        )

    # ---------------------------------------
    # Fasting blood sugar indicator
    # ---------------------------------------
    if patient["fbs"] == 1:
        recommendations.append(
            "Consider checking your blood glucose regularly and discuss elevated readings with a healthcare professional."
        )

    # ---------------------------------------
    # Exercise-induced chest symptoms
    # ---------------------------------------
    if patient["exang"] == 1:
        recommendations.append(
            "Exercise-related chest discomfort or related symptoms should be discussed with a healthcare professional."
        )

    # ---------------------------------------
    # General heart-health recommendations
    # ---------------------------------------
    recommendations.append(
        "Stay physically active according to your abilities and health condition."
    )

    recommendations.append(
        "Choose a balanced diet with plenty of vegetables and fruits and limit excess salt, added sugars and unhealthy fats."
    )

    recommendations.append(
        "Avoid smoking and tobacco products."
    )

    recommendations.append(
        "Avoid harmful alcohol use."
    )

    recommendations.append(
        "Maintain regular health check-ups and monitor important cardiovascular health measurements."
    )

    return recommendations