# CardioRiskAI

## Enhancing Cardio Vascular Health Through Machine Learning

CardioRiskAI is a web-based machine learning application designed to provide preliminary heart health screening using health, lifestyle, and clinical information.

The system provides two screening pathways:

- Basic Screening – uses general health and lifestyle information.
- Clinical Screening – uses clinical measurements and medical test information.

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

The application uses a trained machine learning model to generate a preliminary heart disease screening score.

### 5. Risk Categorization

The result is displayed using categories such as:

- Lower
- Moderate
- High

### 6. Personalized Recommendations

After screening, the system displays recommendations based on the user's submitted information and screening result.

### 7. Screening History

Users can view their previous screening assessments from the dashboard.

### 8. PDF Reports

Users can download a PDF report containing their screening result and recommendations.

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
│
├── .env
├── requirements.txt
│
├── models/
│   └── random_forest.pkl
│
├── ml/
│   ├── risk_assessor.py
│   ├── recommendations.py
│   ├── basic_predictor.py
│   └── basic_recommendations.py
│
├── database/
│   ├── db_connection.py
│   ├── assessment_db.py
│   ├── basic_assessment_db.py
│   └── history_db.py
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── basic_screening.html
│   ├── clinical_screening.html
│   └── history.html
│
├── static/
│   ├── css/
│   │   ├── dashboard.css
│   │   └── screening.css
│   │
│   ├── js/
│   │
│   └── images/
│       └── heart.png
│
└── README.md