# ---------------------------------------
# Risk Assessment Module
# ---------------------------------------

def assess_risk(probability):
    """
    Convert the model probability into a
    project-defined screening category.

    Note:
    These thresholds are prototype thresholds
    for this project and are NOT clinical
    diagnostic cutoffs.
    """

    risk_score = round(probability * 100, 2)

    if risk_score < 30:
        risk_level = "Low"
    elif risk_score < 60:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level
    }