# Proyecto Integrador 5.12 — Predicción de Churn en Telecomunicaciones

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.0-red)
![License](https://img.shields.io/badge/License-Académico-lightgrey)

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Modelos Implementados](#modelos-implementados)
- [Instalación](#instalación)
- [Ejecución de Notebooks](#ejecución-de-notebooks)
- [Ejecutar la API localmente](#ejecutar-la-api-localmente)
- [Docker](#docker)
- [Endpoints de la API](#endpoints-de-la-api)
- [CI/CD con GitHub Actions](#cicd-con-github-actions)
- [Interpretabilidad con LIME](#interpretabilidad-con-lime)
- [Conclusiones](#conclusiones)
- [Referencias](#referencias)

---

## Descripción del Proyecto

El **churn** o abandono de clientes es uno de los problemas más críticos en la industria de telecomunicaciones, con costos de adquisición de nuevos clientes entre 5 y 25 veces superiores a los de retención. Este proyecto implementa un sistema completo de predicción de churn que incluye:

- **Análisis Exploratorio de Datos (EDA)** sobre el dataset IBM Telco Customer Churn.
- **Modelado predictivo** con cuatro algoritmos basados en árboles de decisión: Random Forest, XGBoost, CatBoost y LightGBM.
- **Interpretabilidad local** mediante el método LIME (Local Interpretable Model-agnostic Explanations).
- **API REST** desarrollada con FastAPI para servir predicciones en tiempo real con interfaz web interactiva.
- **Pipeline MLOps** con Docker y GitHub Actions para integración y entrega continua.

El modelo seleccionado (**XGBoost**) alcanza un **AUC-ROC de 0.8481** sobre el conjunto de prueba, identificando correctamente el 51.6% de los clientes que abandonarán la empresa.

---

## Dataset

**IBM Telco Customer Churn**

| Característica | Valor |
|---|---|
| Fuente | IBM Sample Data Sets |
| Registros | 7,043 clientes |
| Variables | 21 (19 predictoras + 1 identificador + 1 objetivo) |
| Variable objetivo | `Churn` (No: 73.46%, Yes: 26.54%) |
| Desbalance de clases | Moderado (~3:1) |

**Variables predictoras:**
- **Demográficas:** gender, SeniorCitizen, Partner, Dependents
- **Servicios:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Contrato:** Contract, PaperlessBilling, PaymentMethod
- **Facturación:** tenure, MonthlyCharges, TotalCharges

---

## Arquitectura del Proyecto

```
MP3/
├── .github/
│   └── workflows/
│       └── ci.yml              # Pipeline CI/CD (lint → test → build Docker)
├── app/
│   └── api.py                  # API FastAPI con interfaz web integrada
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset original IBM
│   └── processed/
│       └── telco_churn_clean.csv                   # Dataset limpio
├── models/
│   ├── best_model.joblib       # Pipeline XGBoost serializado (mejor modelo)
│   └── model_metadata.json     # Metadatos: features, target, parámetros
├── notebooks/
│   ├── 01_eda_telco_churn.ipynb              # Análisis Exploratorio de Datos
│   ├── 02_modeling_pipeline_gridsearch.ipynb # Entrenamiento y comparación de modelos
│   └── 03_interpretability_lime.ipynb        # Interpretabilidad local con LIME
├── reports/
│   ├── feature_importance.csv  # Importancia de variables (XGBoost)
│   ├── model_results.csv       # Métricas comparativas de los 4 modelos
│   └── lime/                   # Explicaciones LIME (HTML + TXT) para 3 perfiles
├── src/                        # Módulos Python reutilizables (extensión futura)
├── tests/
│   └── test_api.py             # Pruebas unitarias con pytest + TestClient
├── Dockerfile                  # Imagen Docker para despliegue
├── requirements.txt            # Dependencias con versiones fijas
└── README.md                   # Este archivo
```

---

## Modelos Implementados

Todos los modelos usan un **Pipeline de scikit-learn** que incluye:
1. Imputación de valores faltantes (`SimpleImputer`)
2. Escalado numérico (`StandardScaler`)
3. Codificación de categóricas (`OneHotEncoder`)
4. Clasificador

La selección de hiperparámetros se realizó con `GridSearchCV` (5-fold StratifiedKFold, `scoring='roc_auc'`, `n_jobs=-1`).

### Tabla Comparativa de Métricas (conjunto de prueba, 20%)

| Modelo | CV AUC | Accuracy | Precision | Recall | F1-Score | **AUC-ROC** |
|--------|--------|----------|-----------|--------|----------|-------------|
| **XGBoost** ⭐ | 0.8504 | **0.8041** | **0.6701** | 0.5160 | 0.5831 | **0.8481** |
| CatBoost | 0.8504 | 0.7480 | 0.5164 | **0.7995** | **0.6275** | 0.8476 |
| LightGBM | 0.8487 | 0.8006 | 0.6598 | 0.5134 | 0.5774 | 0.8474 |
| Random Forest | 0.8453 | 0.7658 | 0.5445 | 0.7193 | 0.6198 | 0.8392 |

**Criterio de selección:** Mayor AUC-ROC en el conjunto de prueba (métrica más robusta ante el desbalance de clases).

### Mejores Hiperparámetros (XGBoost)

```python
{
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1
}
```

### Top 5 Variables más Importantes (XGBoost)

| Variable | Importancia |
|---|---|
| Contract_Month-to-month | 34.73% |
| TechSupport_No | 8.78% |
| OnlineSecurity_No | 8.66% |
| InternetService_Fiber optic | 7.21% |
| InternetService_DSL | 4.68% |

---

## Instalación

### Requisitos previos
- Python 3.10+
- pip
- (Opcional) Docker Desktop

### Instalación local

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd MP3

# 2. Crear entorno virtual (recomendado)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución de Notebooks

Ejecutar los notebooks en el siguiente orden:

```bash
# Iniciar Jupyter
jupyter notebook
```

| Orden | Notebook | Descripción | Prerequisito |
|-------|----------|-------------|--------------|
| 1° | `01_eda_telco_churn.ipynb` | Análisis exploratorio, limpieza, visualizaciones | Dataset raw |
| 2° | `02_modeling_pipeline_gridsearch.ipynb` | Entrenamiento, GridSearch, comparación, guarda modelo | Notebook 01 |
| 3° | `03_interpretability_lime.ipynb` | Explicaciones LIME para 3 perfiles de cliente | Notebook 02 |

> **Nota:** El notebook 02 guarda automáticamente `models/best_model.joblib` y `models/model_metadata.json`, que son necesarios para la API.

---

## Ejecutar la API localmente

```bash
# Desde el directorio raíz del proyecto
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en:
- **Interfaz web:** http://localhost:8000
- **Documentación Swagger:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

---

## Docker

### Construir la imagen

```bash
docker build -t telco-churn-api:latest .
```

### Ejecutar el contenedor

```bash
docker run -d \
  --name telco-churn \
  -p 8000:8000 \
  telco-churn-api:latest
```

### Con variable de entorno personalizada para el modelo

```bash
docker run -d \
  --name telco-churn \
  -p 8000:8000 \
  -e MODEL_PATH=/app/models/best_model.joblib \
  telco-churn-api:latest
```

### Verificar que funciona

```bash
curl http://localhost:8000/health
```

---

## Endpoints de la API

### `GET /`
Interfaz web interactiva con formulario de predicción.

---

### `GET /health`
Verifica el estado del servicio.

**Respuesta:**
```json
{
  "status": "ok",
  "message": "La API se encuentra activa.",
  "model_loaded": true,
  "model_name": "XGBoost",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

---

### `POST /predict`
Predice la probabilidad de churn para un cliente individual.

**Cuerpo de la solicitud:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.50
}
```

**Ejemplo con curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 845.50
  }'
```

**Respuesta:**
```json
{
  "timestamp": "2024-01-15T10:30:00.000000",
  "input_data": { "...": "..." },
  "result": {
    "prediction": 1,
    "prediction_label": "Churn",
    "churn_probability": 0.7532,
    "confidence": "Alta",
    "risk_level": "Alto",
    "risk_factors": [
      "Contrato mes a mes (42.7% de tasa de churn)",
      "Servicio de fibra óptica (41.9% de tasa de churn)",
      "Antigüedad baja (12 meses con la empresa)"
    ],
    "recommendation": "Cliente con alto riesgo de abandono...",
    "threshold_used": 0.5
  }
}
```

---

### `POST /predict-batch`
Predice el churn para múltiples clientes en una sola solicitud.

**Ejemplo con curl:**
```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{"customers": [<cliente1>, <cliente2>]}'
```

---

### `GET /model-info`
Retorna información y métricas del modelo activo.

**Respuesta:**
```json
{
  "model_name": "XGBoost",
  "model_status": "loaded",
  "version": "1.0.0",
  "selection_metric": "test_roc_auc",
  "prediction_threshold_default": 0.5,
  "test_metrics": {
    "test_accuracy": 0.8041,
    "test_precision": 0.6701,
    "test_recall": 0.516,
    "test_f1_score": 0.5831,
    "test_roc_auc": 0.8481,
    "best_cv_roc_auc": 0.8504
  },
  "features_used": ["gender", "SeniorCitizen", "..."]
}
```

---

## CI/CD con GitHub Actions

El pipeline `.github/workflows/ci.yml` se ejecuta automáticamente en cada `push` a `main` o `develop`, y en cada `pull_request` hacia `main`.

```
Push / PR → [Job 1: Lint] → [Job 2: Tests] → [Job 3: Build Docker]
```

| Job | Herramienta | Descripción |
|-----|-------------|-------------|
| **Lint** | flake8 | Verifica estilo PEP8 en `app/api.py` |
| **Test** | pytest | Ejecuta 25+ pruebas unitarias sobre todos los endpoints |
| **Build** | Docker | Construye la imagen y realiza prueba de humo (`/health`) |

---

## Interpretabilidad con LIME

El notebook `03_interpretability_lime.ipynb` aplica **LIME (Local Interpretable Model-agnostic Explanations)** para explicar predicciones individuales.

Se analizan tres perfiles representativos:

| Perfil | Probabilidad de Churn | Churn Real | Predicción |
|--------|----------------------|------------|------------|
| Alta probabilidad | 88.82% | Sí | Sí (VP) |
| Baja probabilidad | 1.05% | No | No (VN) |
| Probabilidad intermedia | 50.01% | Sí | Sí (VP) |

Las explicaciones LIME se guardan en `reports/lime/` en formatos HTML y TXT.

---

## Conclusiones

1. **XGBoost** fue el mejor modelo con **AUC-ROC = 0.8481**, superando a Random Forest, CatBoost y LightGBM.
2. El **tipo de contrato** es la variable más determinante (34.7% de importancia): los clientes con contrato mensual tienen una tasa de churn del 42.7%.
3. La **ausencia de servicios de soporte** (TechSupport, OnlineSecurity) y el **servicio de fibra óptica** son factores de riesgo secundarios pero significativos.
4. El **método de pago** por cheque electrónico está asociado con la mayor tasa de churn (45.3%).
5. LIME confirma que el modelo captura patrones coherentes con el análisis exploratorio: contratos cortos, ausencia de servicios y cargos elevados son señales de riesgo consistentes.

---

## Referencias

1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
3. Prokhorenkova, L. et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS.
4. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD '16.
6. IBM. (2019). *Telco Customer Churn Dataset*. IBM Sample Data Sets.
7. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
