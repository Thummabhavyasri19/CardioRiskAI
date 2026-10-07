import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ---------------------------------------
# 1. Load dataset
# ---------------------------------------
file_path = r"datasets\processed\heart_disease_clean.csv"

df = pd.read_csv(file_path)

X = df.drop("target", axis=1)
y = df["target"]


# ---------------------------------------
# 2. Train-test split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------
# 3. Define models
# ---------------------------------------
models = {

    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ---------------------------------------
# 4. Create models folder
# ---------------------------------------
os.makedirs("models", exist_ok=True)


# ---------------------------------------
# 5. Train, evaluate and save
# ---------------------------------------
for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # Save model
    model_path = f"models/{name}.pkl"

    joblib.dump(model, model_path)

    print("Saved:", model_path)


# ---------------------------------------
# 6. Save feature names
# ---------------------------------------
feature_info = {
    "features": X.columns.tolist()
}

joblib.dump(
    feature_info,
    "models/feature_info.pkl"
)

print("\nFeature information saved:")
print("models/feature_info.pkl")

print("\nAll models trained and saved successfully.")