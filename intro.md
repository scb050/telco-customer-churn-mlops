# Predicción de Churn en Telecomunicaciones

**Proyecto Integrador 5.12 — MLOps**

---

El **churn** o abandono de clientes es uno de los problemas más críticos en telecomunicaciones. Retener un cliente cuesta entre 5 y 25 veces menos que adquirir uno nuevo. Este libro documenta un sistema completo de predicción de churn construido sobre el dataset **IBM Telco Customer Churn** (7 043 clientes, 21 variables).

## Contenido

| Sección | Descripción |
|---|---|
| **Análisis Exploratorio** | Calidad de datos, distribución de churn, variables numéricas y categóricas |
| **Modelado y Comparación** | Pipeline scikit-learn, GridSearchCV, Random Forest · XGBoost · CatBoost · LightGBM |
| **Interpretabilidad LIME** | Explicaciones locales para tres perfiles de cliente |

## Resultados clave

El modelo seleccionado (**XGBoost**) alcanza un **AUC-ROC de 0.8481** en prueba. Las variables más influyentes son el tipo de contrato (mes a mes), la ausencia de soporte técnico y el servicio de fibra óptica.

| Modelo | AUC-ROC | Accuracy | F1 |
|---|---|---|---|
| **XGBoost** ⭐ | **0.8481** | 0.8041 | 0.5831 |
| CatBoost | 0.8476 | 0.7480 | 0.6275 |
| LightGBM | 0.8474 | 0.8006 | 0.5774 |
| Random Forest | 0.8392 | 0.7658 | 0.6198 |

---

## Equipo

| | |
|---|---|
| **Sergio Cadavid** | Análisis exploratorio y preprocesamiento |
| **Juan Conrado** | Modelado, pipeline y API |
| **Mateo Chang** | Interpretabilidad y documentación |

---

*Universidad — 2025*
