
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Seguro Médico & Diabetes — Demo", page_icon="🩺", layout="centered")

ROOT = Path(__file__).resolve().parents[1]
models_dir = ROOT / "models"
reports_dir = ROOT / "reports"

st.title("🩺 Predicción de Costos de Seguro Médico & Riesgo de Diabetes")

tab1, tab2 = st.tabs(["💰 Costos de Seguro Médico", "🧪 Riesgo de Diabetes"])

with tab1:
    st.subheader("Estimador de cargos mensuales")
    reg = joblib.load(models_dir / "insurance_reg.pkl")
    rf_ins = joblib.load(models_dir / "insurance_rf.pkl")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad", 18, 100, 35)
        bmi = st.number_input("BMI", 10.0, 60.0, 28.0, step=0.1)
        children = st.number_input("Número de hijos", 0, 10, 0)
    with col2:
        sex = st.selectbox("Sexo", ["male","female"])
        smoker = st.selectbox("Fumador", ["yes","no"])
        region = st.selectbox("Región", ["southwest","southeast","northwest","northeast"])

    if st.button("Predecir cargo mensual"):
        row = pd.DataFrame([{
            "age": age, "bmi": bmi, "children": children,
            "sex": sex, "smoker": smoker, "region": region
        }])
        pred = reg.predict(row)[0]
        st.success(f"Cargo mensual estimado: ${pred:,.2f}")
    st.caption("Modelo lineal con mejor RMSE tras comparar Linear/Ridge/Lasso. Importancias globales calculadas con RandomForest.")

with tab2:
    st.subheader("Probabilidad de diabetes")
    clf = joblib.load(models_dir / "diabetes_logreg.pkl")
    rf_db = joblib.load(models_dir / "diabetes_rf.pkl")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Embarazos", 0, 20, 1)
        glucose = st.number_input("Glucosa", 0, 300, 120)
        blood_pressure = st.number_input("Presión Diastólica", 0, 200, 70)
        skin_thickness = st.number_input("Espesor de Piel", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulina", 0, 1000, 80)
        bmi = st.number_input("BMI", 0.0, 80.0, 30.0, step=0.1, key='bmi2')
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        age = st.number_input("Edad", 1, 120, 33)

    # Load optimal threshold from report if available
    thr_opt = 0.5
    try:
        import json
        with open(reports_dir / "diabetes_metrics.json") as f:
            meta = json.load(f)
            thr_opt = meta.get("optimal_threshold", 0.5)
    except Exception:
        pass

    th = st.slider("Umbral de decisión", 0.0, 1.0, float(thr_opt), 0.01, help="Valores mayores al umbral se consideran 'diabetes: sí'")

    if st.button("Predecir riesgo"):
        row = pd.DataFrame([{
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age
        }])
        prob = clf.predict_proba(row)[:,1][0]
        pred = int(prob >= th)
        st.info(f"Probabilidad estimada: {prob:.3f} — Predicción: {'Positivo' if pred==1 else 'Negativo'} (umbral={th:.2f})")
    st.caption("Clasificador: Regresión Logística con class_weight='balanced' y estandarización.")

st.markdown("---")
st.write("**Nota:** Esta es una demo educativa. No usar para decisiones médicas reales.")
