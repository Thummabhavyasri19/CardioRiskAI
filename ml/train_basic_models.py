import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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


# =======================================
# 1. File path
# =======================================

file_path = r"datasets\processed\basic_screening_clean.csv"


# =======================================
# 2. Load dataset
# =======================================

print("Loading Basic Screening dataset...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)


# =======================================
# 3. Separate features and target
# =======================================

X = df.drop("target", axis=1)
y = df["target"]


# =======================================
# 4. Train-test split
# =======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# =======================================
# 5. Feature groups
# =======================================

categorical_features = [
    "age_group",
    "sex",
    "smoking",
    "physical_inactivity",
    "high_blood_pressure",
    "diabetes_status",
    "general_health",
    "alcohol_use",
    "previous_stroke"
]

numeric_features = [
    "bmi",
    "sleep_hours"
]


# =======================================
# 6. Logistic Regression preprocessing
# =======================================

logistic_preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),

        (
            "numeric",
            StandardScaler(),
            numeric_features
        )

    ]
)


# =======================================
# 7. Create models
# =======================================

models = {

    "basic_logistic_regression": Pipeline([
        (
            "preprocessor",
            logistic_preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]),

    "basic_decision_tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "basic_random_forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
}


# =======================================
# 8. Create models folder
# =======================================

os.makedirs("models", exist_ok=True)


# =======================================
# 9. Train and evaluate
# =======================================

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------
    # Predictions
    # ---------------------------------------

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------
    # Metrics
    # ---------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # ---------------------------------------
    # Display metrics
    # ---------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # ---------------------------------------
    # Save model
    # ---------------------------------------

    model_path = f"models/{name}.pkl"

    joblib.dump(
        model,
        model_path
    )

    print("Saved:", model_path)


# =======================================
# 10. Save feature information
# =======================================

feature_info = {
    "features": X.columns.tolist(),
    "categorical_features": categorical_features,
    "numeric_features": numeric_features
}

joblib.dump(
    feature_info,
    "models/basic_feature_info.pkl"
)

print("\nFeature information saved:")
print("models/basic_feature_info.pkl")


# =======================================
# 11. Completion
# =======================================

print("\nAll Basic Screening models trained successfully.")