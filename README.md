# Seguro Médico & Diabetes — App (Streamlit)

Demo educativa que entrena dos modelos y despliega una interfaz web:

- **Regresión de costos de seguro médico** (dataset `insurance.csv`).
- **Clasificación de diabetes** (dataset `diabetes.csv`).

## Cómo ejecutar

```bash
#Requerimientos
streamlit
scikit-learn
pandas
numpy
joblib

# 1) Crear entorno
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Entrenar modelos
python scripts/train_insurance.py
python scripts/train_diabetes.py

# 4) Ejecutar la app
streamlit run app/app.py
```

## Respuestas rápidas (se completan tras el entrenamiento)

**1) Umbral ideal (diabetes):** ver `reports/diabetes_metrics.json` → `optimal_threshold`.  
**2) Factores que más influyen en costos de seguro:** ver `reports/insurance_metrics.json` → `top10_factors` (RandomForest).  
**3) Análisis comparativo de características:** En ambos casos se usa **RandomForest** para calcular importancias globales.  
**4) Técnica de optimización que mejoró rendimiento:**  
- Diabetes: balanceo de clases + búsqueda de umbral (Youden) y regularización `C`.  
- Seguro: comparación **Linear vs Ridge vs Lasso** (seleccionado el de menor **RMSE**).  
**5) Contexto de los datos:** `insurance.csv` (edad, sexo, IMC, hijos, fumador, región → predict `charges`). `diabetes.csv` (Pima Indians Diabetes: variables clínicas → predict `Outcome`).  
**6) Análisis de sesgo:** ver sección **Bias** más abajo.

### Métricas (se actualizan tras entrenar)
- Diabetes (AUC): _TBD_
- Diabetes (umbral óptimo): _TBD_
- Seguro (mejor modelo & RMSE): _TBD_

## Bias (resumen)
- **Diabetes**: dataset enfocado en población Pima; contiene variables como `Pregnancies`, por lo que su generalización a varones u otras poblaciones puede ser limitada. Clases desbalanceadas; se usó `class_weight='balanced'` y ajuste de umbral.  
- **Seguro**: la variable `smoker=yes` suele dominar el costo. Hay variables sensibles (`sex`, `region`); revisar diferencias de error por subgrupos antes de uso productivo.

---
