# CardioRiskAI

## Enhancing Cardio Vascular Health Through Machine Learning

CardioRiskAI is a web-based machine learning application designed to provide preliminary heart health screening using health, lifestyle, and clinical information.

The system provides two screening pathways:

- **Basic Screening** – uses general health and lifestyle information.
- **Clinical Screening** – uses clinical measurements and medical test information.

The application provides a screening score, risk category, interpretation, personalized recommendations, screening history, and PDF reports.

> **Disclaimer:** CardioRiskAI provides preliminary screening information and does not provide a medical diagnosis. Users should consult a qualified healthcare professional for medical advice.

---

## Features

### 1. User Authentication

- User registration
- User login
- Session-based authentication
- Logout functionality

### 2. Basic Screening

Users can provide:

- Age
- Gender
- Height
- Weight
- BMI
- Smoking status
- Physical activity
- High blood pressure status
- Diabetes status
- General health
- Sleep hours
- Alcohol use
- Previous stroke history

The system calculates BMI automatically and generates a preliminary screening result.

### 3. Clinical Screening

Users can provide clinical information such as:

- Age
- Gender
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- Oldpeak
- ST segment slope
- Major vessels
- Thalassemia result

### 4. Machine Learning Prediction

The application uses trained machine learning models to generate preliminary heart health screening results.

The project includes models such as:

- Logistic Regression
- Decision Tree
- Random Forest

### 5. Risk Categorization

The screening result is displayed using the following categories:

- **Lower**
- **Moderate**
- **High**

### 6. Personalized Recommendations

After screening, the system provides recommendations based on the submitted health information and screening result.

### 7. Screening History

Users can view their previous screening assessments from the dashboard.

### 8. PDF Reports

Users can download PDF reports containing their screening result and recommendations.

### 9. Dashboard

The dashboard provides access to:

- Basic Screening
- Clinical Screening
- Screening History
- User profile
- Logout

The dashboard also contains a heart health section with an animated heart and pulse/ECG visual.

---

# Technologies Used

## Frontend

- HTML5
- CSS3
- JavaScript
- Jinja2 Templates

## Backend

- Python
- Flask

## Machine Learning

- Pandas
- NumPy
- Scikit-learn
- Joblib

## Database

- Database connection through the project's database modules
- User data
- Assessment data
- Screening history

## Reporting

- ReportLab
- PDF report generation

---

# Project Structure

```text
CardioRiskAI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── database/
│   ├── db_connection.py
│   ├── assessment_db.py
│   ├── basic_assessment_db.py
│   └── history_db.py
│
├── datasets/
│   ├── processed/
│   │   ├── basic_screening_clean.csv
│   │   └── heart_disease_clean.csv
│   │
│   └── raw/
│       └── heart+disease/
│           └── raw heart disease dataset files
│
├── ml/
│   ├── basic_predictor.py
│   ├── basic_recommendations.py
│   ├── inspect_brfss.py
│   ├── preprocess.py
│   ├── preprocess_brfss.py
│   ├── recommendations.py
│   ├── risk_assessor.py
│   ├── test_basic_model.py
│   ├── test_model.py
│   ├── train_basic_models.py
│   ├── train_models.py
│   └── tune_basic_model.py
│
├── models/
│   ├── basic_decision_tree.pkl
│   ├── basic_feature_info.pkl
│   ├── basic_logistic_regression.pkl
│   ├── basic_logistic_tuned.pkl
│   ├── basic_random_forest.pkl
│   ├── decision_tree.pkl
│   ├── feature_info.pkl
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── static/
│   ├── css/
│   │   ├── auth.css
│   │   ├── dashboard.css
│   │   └── screening.css
│   │
│   └── images/
│       └── heart.png
│
└── templates/
    ├── index.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── basic_screening.html
    ├── clinical_screening.html
    ├── history.html
    └── report.html
