"""
Pruebas unitarias para la API de Predicción de Churn.
Proyecto Integrador 5.12 - Capítulo 5 Machine Learning

Ejecutar con: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

# Asegurar que el directorio raíz esté en el path para importar app.api
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

# ─────────────────────────────────────────────
# Datos de ejemplo para pruebas
# ─────────────────────────────────────────────

CLIENTE_VALIDO = {
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
    "TotalCharges": 845.5
}

CLIENTE_ALTO_RIESGO = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 99.65,
    "TotalCharges": 99.65
}

CLIENTE_BAJO_RIESGO = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.25,
    "TotalCharges": 2715.0
}


# ─────────────────────────────────────────────
# Test: GET /health
# ─────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_model_loaded(self):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True

    def test_health_has_version(self):
        response = client.get("/health")
        data = response.json()
        assert "version" in data

    def test_health_has_model_name(self):
        response = client.get("/health")
        data = response.json()
        assert "model_name" in data


# ─────────────────────────────────────────────
# Test: GET /
# ─────────────────────────────────────────────

class TestHome:
    def test_home_status(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_home_is_html(self):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_home_contains_title(self):
        response = client.get("/")
        assert "Telco" in response.text


# ─────────────────────────────────────────────
# Test: POST /predict
# ─────────────────────────────────────────────

class TestPredict:
    def test_predict_status_ok(self):
        response = client.post("/predict", json=CLIENTE_VALIDO)
        assert response.status_code == 200

    def test_predict_has_required_fields(self):
        response = client.post("/predict", json=CLIENTE_VALIDO)
        data = response.json()
        result = data["result"]
        assert "churn_probability" in result
        assert "prediction" in result
        assert "prediction_label" in result
        assert "risk_level" in result
        assert "confidence" in result
        assert "recommendation" in result

    def test_predict_probability_range(self):
        response = client.post("/predict", json=CLIENTE_VALIDO)
        result = response.json()["result"]
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_predict_binary_prediction(self):
        response = client.post("/predict", json=CLIENTE_VALIDO)
        result = response.json()["result"]
        assert result["prediction"] in [0, 1]

    def test_predict_label_values(self):
        response = client.post("/predict", json=CLIENTE_VALIDO)
        result = response.json()["result"]
        assert result["prediction_label"] in ["Churn", "No Churn"]

    def test_predict_risk_level_values(self):
        response = client.post("/predict", json=CLIENTE_VALIDO)
        result = response.json()["result"]
        assert result["risk_level"] in ["Bajo", "Medio", "Alto"]

    def test_predict_confidence_values(self):
        response = client.post("/predict", json=CLIENTE_VALIDO)
        result = response.json()["result"]
        assert result["confidence"] in ["Alta", "Media", "Baja"]

    def test_predict_high_risk_customer(self):
        response = client.post("/predict", json=CLIENTE_ALTO_RIESGO)
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["churn_probability"] > 0.5

    def test_predict_low_risk_customer(self):
        response = client.post("/predict", json=CLIENTE_BAJO_RIESGO)
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["churn_probability"] < 0.5

    def test_predict_invalid_gender(self):
        datos_invalidos = {**CLIENTE_VALIDO, "gender": "Otro"}
        response = client.post("/predict", json=datos_invalidos)
        assert response.status_code == 422

    def test_predict_invalid_contract(self):
        datos_invalidos = {**CLIENTE_VALIDO, "Contract": "Weekly"}
        response = client.post("/predict", json=datos_invalidos)
        assert response.status_code == 422

    def test_predict_negative_tenure(self):
        datos_invalidos = {**CLIENTE_VALIDO, "tenure": -5}
        response = client.post("/predict", json=datos_invalidos)
        assert response.status_code == 422

    def test_predict_missing_required_field(self):
        datos_incompletos = {k: v for k, v in CLIENTE_VALIDO.items() if k != "tenure"}
        response = client.post("/predict", json=datos_incompletos)
        assert response.status_code == 422

    def test_predict_with_custom_threshold(self):
        response = client.post("/predict?threshold=0.3", json=CLIENTE_VALIDO)
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["threshold_used"] == 0.3

    def test_predict_has_timestamp(self):
        response = client.post("/predict", json=CLIENTE_VALIDO)
        data = response.json()
        assert "timestamp" in data

    def test_predict_has_risk_factors_list(self):
        response = client.post("/predict", json=CLIENTE_VALIDO)
        result = response.json()["result"]
        assert "risk_factors" in result
        assert isinstance(result["risk_factors"], list)


# ─────────────────────────────────────────────
# Test: POST /predict-batch
# ─────────────────────────────────────────────

class TestPredictBatch:
    def test_batch_predict_status(self):
        payload = {"customers": [CLIENTE_VALIDO, CLIENTE_ALTO_RIESGO]}
        response = client.post("/predict-batch", json=payload)
        assert response.status_code == 200

    def test_batch_predict_count(self):
        payload = {"customers": [CLIENTE_VALIDO, CLIENTE_ALTO_RIESGO, CLIENTE_BAJO_RIESGO]}
        response = client.post("/predict-batch", json=payload)
        data = response.json()
        assert data["total_customers"] == 3
        assert len(data["results"]) == 3


# ─────────────────────────────────────────────
# Test: GET /model-info
# ─────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_status(self):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_model_name(self):
        response = client.get("/model-info")
        data = response.json()
        assert "model_name" in data

    def test_model_info_has_status(self):
        response = client.get("/model-info")
        data = response.json()
        assert data["model_status"] == "loaded"

    def test_model_info_has_metrics(self):
        response = client.get("/model-info")
        data = response.json()
        assert "test_metrics" in data

    def test_model_info_has_features(self):
        response = client.get("/model-info")
        data = response.json()
        assert "features_used" in data
        assert len(data["features_used"]) > 0
