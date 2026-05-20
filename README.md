# Customer Churn Prediction using XGBoost

## Project Overview

This project aims to predict customer churn in a telecommunications company using Machine Learning. Customer churn refers to customers who stop using the company's services.

The model was developed using the XGBoost algorithm and deployed as an interactive web application using Streamlit and Docker.

---

# Business Problem

Customer churn can significantly reduce company revenue and increase customer acquisition costs.

This project helps businesses:
- identify customers with high churn risk
- understand factors influencing churn
- improve retention strategies
- reduce customer loss

---

# Dataset

Dataset: IBM Telco Customer Churn Dataset

The dataset contains customer information such as:
- customer demographics
- subscription services
- contract types
- payment methods
- monthly charges
- churn status

Target variable:
- `Churn`
  - Yes → customer churned
  - No → customer stayed

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Docker
- Matplotlib
- Seaborn

---

# Machine Learning Workflow

## 1. Data Understanding
- understand data structures
- check for missing values
- check for duplicate data

## 2. Exploratory Data Analysis (EDA)
Performed visual analysis to understand:
- churn distribution
- tenure vs churn
- monthly charges vs churn
- contract type impact
- internet service impact
- payment method behavior
- see the correlation between features

## 3. Data Preprocessing

- one-hot encoding
- feature scaling using StandardScaler
- train-test split
- handling missing values
- converting data types
- removing unnecessary columns

## 4. Modeling
Models used:
- Logistic Regression
- XGBoost Classifier

## 5. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cross Validation

---

# Model Performance

he project evaluated two machine learning models: Logistic Regression and XGBoost Classifier.

## Logistic Regression
- Accuracy: 0.73
- Precision: 0.50
- Recall: 0.78
- F1-Score: 0.61
- ROC-AUC: 0.83

## XGBoost Classifier
- Accuracy: 0.74
- Precision: 0.51
- Recall: 0.80
- F1-Score: 0.62
- ROC-AUC: 0.83

XGBoost achieved the best overall performance, particularly in recall score, which is important for identifying customers with high churn risk.

---

# Key Insights

1. Customers with shorter tenure periods are more likely to churn.

2. Customers with month-to-month contracts exhibit significantly higher churn rates.

3. Higher monthly charges are associated with increased churn behavior.

4. Customers without technical support tend to churn more frequently.

5. Customers using fiber optic internet and electronic check payment methods show relatively higher churn tendencies.

---

# Business Recommendations

- Encourage customers to switch to long-term contracts through special promotions and benefits.
- Improve onboarding and loyalty programs for new customers with short tenure.
- Improving the quality of technical support to increase customer retention.
- Provide promos or special offers to customers with high monthly charges.

---

# Streamlit Application

The project includes an interactive Streamlit application where users can:
- input customer information
- predict churn probability
- receive business recommendations

---

# Docker Implementation

The application was containerized using Docker to:
- ensure environment consistency
- simplify deployment
- improve portability

---

# Project Structure

```bash
churn-prediction/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── xgboost_churn_model.pkl
├── scaler.pkl
├── model_columns.pkl
├── customer-churn-prediction.ipynb
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/farisazizy289/customer-churn-prediction.git
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit App

```bash
streamlit run app.py
```

---

# Run with Docker

## Build Docker Image

```bash
docker build -t churn-app .
```

## Run Docker Container

```bash
docker run -p 8501:8501 churn-app
```

Application will run at:

```bash
http://localhost:8501
```

---

# Deployment

This application can be deployed using:
- Streamlit Community Cloud
- Render
- Railway
- Google Cloud Run

---

# Future Improvements

- Hyperparameter tuning
- SMOTE for imbalance handling
- Explainable AI (SHAP)
- FastAPI backend implementation
- Cloud deployment optimization

---

# Author

Faris Ahmad Rizky Azizy

---

# Live Demo

https://customer-churn-prediction.streamlit.app
