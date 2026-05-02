import streamlit as st
import xgboost as xgb
import pandas as pd
import altair as alt
import os
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


st.set_page_config(page_title="💳 Credit Default Detector", layout="wide")

st.title("💳 Credit Default Detection System")
st.markdown("Choose between full model (detailed) or fast model (top 10 features).")

# --- Mode Selection ---
mode = st.radio("🧠 Choose Prediction Mode", ["Full Mode ", "Fast Mode "])
is_fast = "Fast" in mode

model_path_full = "C:/pyhton/Lab-task-01/research/credit_model.json"
model_path_fast = "C:/pyhton/Lab-task-01/research/model_fast.json"
scaler_path = "C:/pyhton/Lab-task-01/research/scaler_full.json" 

# --- Load Model ---
model = xgb.Booster()
if is_fast:
    if not os.path.exists(model_path_fast):
        st.error("❌ Fast model not found!")
        st.stop()
    model.load_model(model_path_fast)
else:
    if not os.path.exists(model_path_full):
        st.error("❌ Full model not found!")
        st.stop()
    model.load_model(model_path_full)

# --- Load Scaler for Fast Mode ---
if is_fast:
    try:
        with open(scaler_path, "r") as f:
            scaler_json = json.load(f)
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_json['mean'])
            scaler.scale_ = np.array(scaler_json['scale'])
    except:
        st.error("⚠️ Failed to load scaler for fast mode.")
        st.stop()

# --- Custom Style ---
st.markdown("""
<style>
    .title { font-size: 36px; font-weight: bold; color: #56C8D8; }
    .subtitle { font-size: 20px; color: #DDDDDD; }
    .section { background-color: #1e1e1e; padding: 1.5em; border-radius: 12px; box-shadow: 2px 2px 12px #00000040; }
    .metric { color: #00FFAA; font-weight: bold; font-size: 20px; }
    .bar { height: 14px; background-color: #222; border-radius: 4px; overflow: hidden; }
    .bar-fill { height: 14px; background: linear-gradient(to right, #00FFA3, #007AFF); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# --- Title & Description ---
st.markdown('<div class="title">🎯 Credit Default Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Using Machine Learning (XGBoost)</div>', unsafe_allow_html=True)
st.markdown("___")

# --- Sidebar Inputs ---
st.sidebar.markdown("## 🧾 Client Info")
threshold = st.sidebar.slider("🎚️ Risk Threshold", 0.0, 1.0, 0.5, step=0.01)

features_fast = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'LIMIT_BAL', 'EDUCATION', 'AVG_PAY_AMT', 'AGE']

def user_input():
    input_data = {}

    with st.sidebar.expander("👤 Personal Info", expanded=True):
        input_data["LIMIT_BAL"] = st.slider("💰 Credit Limit", 0, 1000000, 20000, step=1000)
        input_data["EDUCATION"] = st.selectbox("🎓 Education", [1, 2, 3, 4], format_func=lambda x: ["Graduate", "University", "High School", "Others"][x - 1])
        input_data["AGE"] = st.slider("🎂 Age", 18, 100, 30)

        if not is_fast:
            input_data["SEX"] = st.radio("👨‍🦰 Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            input_data["MARRIAGE"] = st.radio("💍 Marital Status", [1, 2, 3], format_func=lambda x: ["Married", "Single", "Others"][x - 1])

    with st.sidebar.expander("📉 Payment History"):
        for i in [0, 2, 3, 4, 5, 6]:
            input_data[f"PAY_{i}"] = st.number_input(f"PAY_{i}", -2, 8, 0)

    if not is_fast:
        with st.sidebar.expander("📄 Billing Info"):
            for i in range(6):
                input_data[f"BILL_AMT{i+1}"] = st.number_input(f"BILL_AMT{i+1}", value=0.0)

    with st.sidebar.expander("💸 Payment Amounts"):
        for i in range(6):
            input_data[f"PAY_AMT{i+1}"] = st.number_input(f"PAY_AMT{i+1}", value=0.0)

    return input_data

def predict_single(input_dict):
    if is_fast:
        avg_amt = sum([input_dict[f"PAY_AMT{i+1}"] for i in range(6)]) / 6
        data = {
            "PAY_0": input_dict["PAY_0"],
            "PAY_2": input_dict["PAY_2"],
            "PAY_3": input_dict["PAY_3"],
            "PAY_4": input_dict["PAY_4"],
            "PAY_5": input_dict["PAY_5"],
            "PAY_6": input_dict["PAY_6"],
            "LIMIT_BAL": input_dict["LIMIT_BAL"],
            "EDUCATION": input_dict["EDUCATION"],
            "AGE": input_dict["AGE"],
            "AVG_PAY_AMT": avg_amt
        }
        df = pd.DataFrame([data])[features_fast]
        df_scaled = scaler.transform(df)
        dmatrix = xgb.DMatrix(df_scaled, feature_names=features_fast)
    else:
        df = pd.DataFrame([input_dict])
        dmatrix = xgb.DMatrix(df)
    prediction = model.predict(dmatrix)
    return prediction[0]

def predict_batch(df):
    if is_fast:
        df["AVG_PAY_AMT"] = df[[f"PAY_AMT{i+1}" for i in range(6)]].mean(axis=1)
        df_fast = df[features_fast]
        df_scaled = scaler.transform(df_fast)
        dmatrix = xgb.DMatrix(df_scaled, feature_names=features_fast)
    else:
        dmatrix = xgb.DMatrix(df)
    preds = model.predict(dmatrix)
    df['Prediction_Prob'] = preds
    df['Risk'] = (preds >= threshold).astype(int)
    return df


# --- Single Prediction ---
st.markdown("### 🧠 Single Client Prediction")
inputs = user_input()

if st.button("🚨 Predict Now", type="primary"):
    prob = predict_single(inputs)
    risk = int(prob >= threshold)

    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown("#### 📊 Model Output")
    st.markdown(f"<div class='metric'>Confidence: {prob*100:.2f}%</div>", unsafe_allow_html=True)
    st.markdown('<div class="bar"><div class="bar-fill" style="width: {}%;"></div></div>'.format(prob*100), unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Risk Type": ["Default", "No Default"],
        "Probability": [prob, 1 - prob]
    })
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Risk Type:N", title=None),
        y=alt.Y("Probability:Q", title="Probability"),
        color=alt.Color("Risk Type:N", scale=alt.Scale(range=["#FF5252", "#00E676"]))
    ).properties(height=200)
    st.altair_chart(chart, use_container_width=True)

    if risk:
        st.error("❌ High Risk: Likely to Default")
        st.image("https://media.giphy.com/media/l0Exk8EUzSLsrErEQ/giphy.gif", width=500)
    else:
        st.success("✅ Low Risk: Unlikely to Default")
        st.image("https://media.giphy.com/media/OkJat1YNdoD3W/giphy.gif", width=500)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Batch Prediction ---
st.markdown("---")
st.markdown("### 📂 Bulk Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = [
            "LIMIT_BAL", "EDUCATION", "AGE", "PAY_0", "PAY_2", "PAY_3",
            "PAY_4", "PAY_5", "PAY_6"
        ] + ([f"PAY_AMT{i+1}" for i in range(6)] if is_fast else [
            "SEX", "MARRIAGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
            "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
        ])

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing columns in CSV: {missing_cols}")
        else:
            results = predict_batch(df)
            st.success("✅ Predictions Complete")
            st.dataframe(results.head())

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
