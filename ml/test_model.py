import pandas as pd
import joblib


# ---------------------------------------
# 1. Load saved model
# ---------------------------------------
model = joblib.load("models/random_forest.pkl")

# ---------------------------------------
# 2. Create sample patient
# ---------------------------------------
patient = pd.DataFrame([{
    "age": 45,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 220,
    "fbs": 0,
    "restecg": 0,
    "thalach": 165,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3
}])


# ---------------------------------------
# 3. Make prediction
# ---------------------------------------
prediction = model.predict(patient)[0]

probability = model.predict_proba(patient)[0][1]


# ---------------------------------------
# 4. Display result
# ---------------------------------------
print("Patient Information:")
print(patient.to_string(index=False))

print("\nPrediction:", prediction)

print(
    "Probability of positive class:",
    round(probability * 100, 2),
    "%"
)


if prediction == 1:
    print("Result: Positive screening prediction")
else:
    print("Result: Negative screening prediction")