# Proyecto Predicción de Churn en Telecomunicaciones

## Integrantes

- **Mateo Chang**
- **Sergio Cadavid**
- **Juan Camilo Conrado**

---

## Contexto del Proyecto

En un entorno empresarial altamente competitivo, la capacidad de anticipar el abandono de clientes constituye una ventaja estratégica para las compañías de telecomunicaciones. Las decisiones relacionadas con retención, segmentación comercial, calidad del servicio, fidelización y campañas comerciales dependen en gran medida de la capacidad de analizar datos históricos y convertirlos en información útil para la toma de decisiones.

Este proyecto tiene como objetivo construir una solución analítica basada en Machine Learning para predecir la probabilidad de abandono de clientes, también conocida como **churn**. A partir de información demográfica, contractual, de servicios y facturación, se busca identificar patrones asociados al abandono de clientes y apoyar la generación de estrategias de retención más efectivas.

La presente entrega integra el desarrollo completo del flujo de trabajo de aprendizaje automático: análisis exploratorio de datos, preprocesamiento, modelado predictivo, evaluación de modelos, interpretación de predicciones y diseño de una arquitectura básica de despliegue mediante FastAPI.

---

## Objetivo General

Construir un modelo de clasificación supervisada que permita estimar la probabilidad de abandono de clientes en una empresa de telecomunicaciones, utilizando algoritmos basados en árboles de decisión e incorporando herramientas de interpretabilidad y despliegue.

---

## Objetivos Específicos

- Realizar un análisis exploratorio de datos para comprender la estructura, calidad y comportamiento del dataset.
- Identificar variables relevantes asociadas al abandono de clientes.
- Implementar pipelines de preprocesamiento para variables numéricas y categóricas.
- Entrenar y comparar modelos como Random Forest, XGBoost, CatBoost y LightGBM.
- Ajustar hiperparámetros mediante GridSearchCV y validación cruzada estratificada.
- Evaluar el desempeño de los modelos mediante métricas como accuracy, precision, recall, F1-score y ROC AUC.
- Interpretar predicciones individuales utilizando LIME.
- Diseñar una API con FastAPI para servir predicciones del modelo entrenado.
- Preparar una estructura básica de MLOps con archivos de soporte para Docker, pruebas automáticas y GitHub Actions.

---

## Descripción del Dataset

El dataset utilizado corresponde a **Telco Customer Churn**, una base de datos ampliamente utilizada para problemas de clasificación binaria en el contexto de telecomunicaciones.

La variable objetivo del proyecto es:

- **Churn**: indica si un cliente abandonó la empresa (`Yes`) o permaneció en ella (`No`).

El conjunto de datos contiene información relacionada con:

- Características demográficas del cliente.
- Servicios contratados.
- Tipo de contrato.
- Método de pago.
- Cargos mensuales.
- Cargos totales acumulados.
- Permanencia del cliente en la empresa.
- Estado final de abandono o permanencia.

---

## Metodología del Proyecto

El desarrollo del proyecto se organiza en tres etapas principales.

### 1. Análisis Exploratorio de Datos

Se revisa la estructura del dataset, los tipos de variables, valores faltantes, duplicados, distribución de la variable objetivo y patrones relevantes relacionados con el churn. Esta fase permite comprender el comportamiento de los datos antes de construir los modelos predictivos.

### 2. Modelado Predictivo

Se construyen modelos de clasificación utilizando `Pipeline`, `GridSearchCV` y validación cruzada estratificada. Los modelos evaluados incluyen Random Forest, XGBoost, CatBoost y LightGBM.

El objetivo de esta etapa es comparar el rendimiento de diferentes algoritmos y seleccionar el modelo con mejor capacidad predictiva.

### 3. Interpretabilidad del Modelo

Se utiliza LIME para explicar predicciones individuales del modelo seleccionado. Esto permite comprender qué variables influyen en la predicción de churn para clientes específicos, facilitando una interpretación más clara desde una perspectiva de negocio.

---

## Arquitectura General del Proyecto

El proyecto se organiza en una estructura reproducible:

```text
MP3/
│
├── app/
│   └── api.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
│
├── notebooks/
│   ├── 01_eda_telco_churn.ipynb
│   ├── 02_modeling_pipeline_gridsearch.ipynb
│   └── 03_interpretability_lime.ipynb
│
├── reports/
│   ├── feature_importance.csv
│   ├── model_results.csv
│   └── lime/
│
├── src/
├── tests/
├── Dockerfile
├── requirements.txt
├── README.md
├── _config.yml
├── _toc.yml
└── intro.md