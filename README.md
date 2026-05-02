# 💳 Credit Default Prediction System (Machine Learning)

## 📌 Overview

This project implements a machine learning-based system to predict whether a customer is likely to default on credit payments. It uses financial and 
behavioral features to classify users into risk categories and provides real-time predictions through an interactive web interface.
The system is designed to demonstrate practical application of machine learning in financial risk analysis.

---

## 🎯 Objectives

* Predict credit default risk using real-world features
* Compare model performance and optimize for accuracy vs speed
* Build an interactive interface for real-time prediction

---

## ⚙️ Features

* Machine learning model using XGBoost
* Two prediction modes:

  * **Full Model** → Higher accuracy using more features
  * **Fast Model** → Faster predictions with fewer features
* Data preprocessing and feature scaling
* Real-time prediction via Streamlit interface
* Confidence score visualization

---

## 🧠 Machine Learning Approach

* Problem Type: **Binary Classification**
* Algorithm: **XGBoost**
* Preprocessing:

  * Feature scaling using StandardScaler
  * Handling missing/structured input data
* Evaluation:

  * Accuracy-based comparison between models

---

## 🏗️ Project Structure

```id="ml1"
credit-default-prediction/
│
├── credit_app.py        # Streamlit application
├── credit_model.json    # Trained full model
├── model_fast.json      # Optimized model
├── scaler_full.json     # Saved scaler parameters
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```id="ml2"
git clone
```

### 2. Install dependencies

```id="ml3"
pip install -r requirements.txt
```

### 3. Run the application

```id="ml4"
streamlit run credit_app.py
```

---

## 📊 How It Works

```id="mlflow"
User Input → Feature Scaling → ML Model → Prediction → Output (Risk + Confidence)
```

* User enters financial details
* Input is scaled using saved preprocessing parameters
* Model predicts default probability
* System outputs:

  * Risk classification
  * Confidence score

---

## 📈 Example Use Case

* Credit risk assessment for financial institutions
* Loan approval support systems
* Risk analysis for customer behavior

---

## ⚠️ Limitations

* Uses pre-trained model (no dynamic training in app)
* Simplified feature set for demonstration

---

## 🔮 Future Improvements

* Improve evaluation metrics (ROC-AUC, F1-score)
* Deploy as web service (Flask / FastAPI)
* Add explainability (SHAP / feature importance visualization)

---

## 👤 Author

Shahriar Kobir Sabbir

---
