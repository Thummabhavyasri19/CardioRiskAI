import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# =======================================
# 1. Load dataset
# =======================================

file_path = r"datasets\processed\basic_screening_clean.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)


# =======================================
# 2. Separate features and target
# =======================================

X = df.drop("target", axis=1)
y = df["target"]


# =======================================
# 3. First split
# =======================================
# 80% development data
# 20% untouched final test data

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =======================================
# 4. Second split
# =======================================
# 80% training
# 20% validation

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    random_state=42,
    stratify=y_dev
)


print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Final test:", X_test.shape)


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
# 6. Preprocessor
# =======================================

preprocessor = ColumnTransformer(
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
# 7. Logistic Regression model
# =======================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# =======================================
# 8. Train
# =======================================

print("\nTraining Logistic Regression...")

model.fit(
    X_train,
    y_train
)


# =======================================
# 9. Validation probabilities
# =======================================

validation_probability = model.predict_proba(
    X_val
)[:, 1]


print("\nValidation ROC-AUC:")
print(
    round(
        roc_auc_score(
            y_val,
            validation_probability
        ),
        4
    )
)

print("\nValidation PR-AUC:")
print(
    round(
        average_precision_score(
            y_val,
            validation_probability
        ),
        4
    )
)


# =======================================
# 10. Find threshold
# =======================================
# We prefer recall >= 0.80,
# then choose the threshold producing
# the highest precision.

best_threshold = None
best_precision = -1
best_recall = 0
best_f1 = 0


for threshold in [
    x / 100
    for x in range(10, 91)
]:

    validation_prediction = (
        validation_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        validation_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        validation_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        validation_prediction,
        zero_division=0
    )

    if recall >= 0.80:

        if precision > best_precision:

            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1


# =======================================
# 11. Display selected threshold
# =======================================

print("\n=======================================")
print("SELECTED THRESHOLD")
print("=======================================")

print("Threshold:", best_threshold)
print("Validation Precision:", round(best_precision, 4))
print("Validation Recall:", round(best_recall, 4))
print("Validation F1:", round(best_f1, 4))


# =======================================
# 12. Final test evaluation
# =======================================

test_probability = model.predict_proba(
    X_test
)[:, 1]


test_prediction = (
    test_probability >= best_threshold
).astype(int)


accuracy = accuracy_score(
    y_test,
    test_prediction
)

precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probability
)

pr_auc = average_precision_score(
    y_test,
    test_probability
)


# =======================================
# 13. Final results
# =======================================

print("\n=======================================")
print("FINAL TEST RESULTS")
print("=======================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


# =======================================
# 14. Confusion matrix
# =======================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_prediction
    )
)


# =======================================
# 15. Save tuned model information
# =======================================

model_info = {
    "model": model,
    "threshold": best_threshold,
    "features": X.columns.tolist(),
    "categorical_features": categorical_features,
    "numeric_features": numeric_features
}


joblib.dump(
    model_info,
    "models/basic_logistic_tuned.pkl"
)


print("\nSaved:")
print("models/basic_logistic_tuned.pkl")