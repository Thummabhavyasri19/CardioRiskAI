import pandas as pd
import joblib


# =======================================
# 1. Load tuned Basic Screening model
# =======================================

model_info = joblib.load(
    "models/basic_logistic_tuned.pkl"
)

model = model_info["model"]
threshold = model_info["threshold"]


print("Basic Screening model loaded successfully")
print("Decision threshold:", threshold)


# =======================================
# 2. Example Patient
# =======================================
# Example: young adult without known
# major risk factors

patient = pd.DataFrame([{
    "age_group": 1,
    "sex": 1,
    "bmi": 21.5,
    "smoking": 0,
    "physical_inactivity": 0,
    "high_blood_pressure": 0,
    "diabetes_status": 3,
    "general_health": 2,
    "sleep_hours": 7,
    "alcohol_use": 0,
    "previous_stroke": 0
}])


# =======================================
# 3. Get model probability
# =======================================

probability = model.predict_proba(
    patient
)[0][1]


model_score = round(
    probability * 100,
    2
)


# =======================================
# 4. Apply tuned threshold
# =======================================

prediction = int(
    probability >= threshold
)


# =======================================
# 5. Display patient information
# =======================================

print("\nPatient Information:")
print(
    patient.to_string(index=False)
)


print("\nModel probability score:",
      model_score, "%")

print("Classification:",
      prediction)


# =======================================
# 6. Display interpretation
# =======================================

if prediction == 1:

    print(
        "Result: Positive basic screening signal"
    )

else:

    print(
        "Result: Negative basic screening signal"
    )