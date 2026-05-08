"""
API de Predicción de Churn en Telecomunicaciones
Proyecto Integrador 5.12 - Capítulo 5 Machine Learning
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ============================================================
# Configuración de logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("telco_churn_api")

# ============================================================
# Rutas de archivos
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
RESULTS_PATH = BASE_DIR / "reports" / "model_results.csv"

# ============================================================
# Carga de modelo y artefactos al iniciar
# ============================================================

model = joblib.load(MODEL_PATH)
logger.info(f"Modelo cargado desde: {MODEL_PATH}")

try:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info("Metadatos del modelo cargados.")
except FileNotFoundError:
    metadata = {}
    logger.warning("Metadatos no encontrados; se usarán valores por defecto.")

try:
    results_df = pd.read_csv(RESULTS_PATH)
    _model_name = metadata.get("best_model_name", "XGBoost")
    _mask = results_df["model"] == _model_name
    best_model_metrics = results_df[_mask].iloc[0].to_dict() if _mask.any() else {}
    logger.info("Métricas del modelo cargadas.")
except Exception:
    best_model_metrics = {}
    logger.warning("No se pudieron cargar las métricas desde el CSV.")

# ============================================================
# Metadatos de documentación de la API
# ============================================================

tags_metadata = [
    {
        "name": "General",
        "description": "Endpoints para verificar el estado del servicio y la página principal."
    },
    {
        "name": "Predicción",
        "description": "Endpoints para estimar la probabilidad de churn de uno o varios clientes."
    },
    {
        "name": "Modelo",
        "description": "Información sobre el modelo activo: métricas de desempeño y variables usadas."
    }
]

# ============================================================
# Inicialización de FastAPI
# ============================================================

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="""
## API de Predicción de Churn en Clientes de Telecomunicaciones

Esta API estima la probabilidad de abandono de clientes usando el modelo **XGBoost**
entrenado con el dataset IBM Telco Customer Churn (7,043 clientes, 19 features).

### Modelos evaluados
| Modelo | AUC-ROC Test |
|--------|-------------|
| **XGBoost** ⭐ | **0.8481** |
| CatBoost | 0.8476 |
| LightGBM | 0.8474 |
| Random Forest | 0.8392 |

### Endpoints
- `GET /` — Interfaz web interactiva con formulario completo
- `GET /health` — Estado del servicio
- `POST /predict` — Predicción individual con factores de riesgo
- `POST /predict-batch` — Predicción por lotes
- `GET /model-info` — Métricas del modelo activo
    """,
    version="1.0.0",
    contact={
        "name": "Proyecto ML Capítulo 5 - Telco Churn",
        "email": "proyecto.ml@universidad.edu"
    },
    openapi_tags=tags_metadata
)

# ============================================================
# Middleware CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Esquemas de entrada (Pydantic)
# ============================================================

class CustomerData(BaseModel):
    gender: Literal["Female", "Male"] = Field(
        ..., description="Género del cliente"
    )
    SeniorCitizen: int = Field(
        ..., ge=0, le=1
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Si el cliente tiene pareja"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Si el cliente tiene dependientes"
    )
    tenure: int = Field(
        ..., ge=0
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Si tiene servicio telefónico"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Si tiene múltiples líneas telefónicas"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Tipo de servicio de internet"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Si tiene seguridad en línea contratada"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Si tiene respaldo en línea contratado"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Si tiene protección de dispositivos"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Si tiene soporte técnico contratado"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Si tiene servicio de TV por streaming"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Si tiene servicio de películas por streaming"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Tipo de contrato del cliente"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Si usa facturación electrónica"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(..., description="Método de pago")
    MonthlyCharges: float = Field(
        ..., ge=0
    )
    TotalCharges: float = Field(
        ..., ge=0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35, "TotalCharges": 845.5
            }
        }
    }


class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

# ============================================================
# Esquemas de salida
# ============================================================

class PredictionResult(BaseModel):
    prediction: int
    prediction_label: str
    churn_probability: float
    confidence: str
    risk_level: str
    risk_factors: List[str]
    recommendation: str
    threshold_used: float


class PredictionResponse(BaseModel):
    timestamp: str
    input_data: CustomerData
    result: PredictionResult


class BatchPredictionResponse(BaseModel):
    timestamp: str
    total_customers: int
    results: List[PredictionResult]

# ============================================================
# Funciones auxiliares
# ============================================================

def customer_to_dataframe(customer: CustomerData) -> pd.DataFrame:
    try:
        data = customer.model_dump()
    except AttributeError:
        data = customer.dict()
    return pd.DataFrame([data])


def classify_risk(probability: float) -> str:
    if probability >= 0.70:
        return "Alto"
    elif probability >= 0.40:
        return "Medio"
    return "Bajo"


def classify_confidence(probability: float) -> str:
    distance = abs(probability - 0.5)
    if distance >= 0.25:
        return "Alta"
    elif distance >= 0.10:
        return "Media"
    return "Baja"


def get_risk_factors(customer: CustomerData, probability: float) -> List[str]:
    """Identifica los factores de riesgo presentes en el cliente."""
    if probability < 0.40:
        return []

    factors = []

    if customer.Contract == "Month-to-month":
        factors.append("Contrato mes a mes (42.7% tasa de churn histórica)")

    if customer.InternetService == "Fiber optic":
        factors.append("Servicio de fibra óptica (41.9% tasa de churn histórica)")

    if customer.tenure <= 12:
        factors.append(f"Antigüedad baja ({customer.tenure} meses con la empresa)")

    has_internet = customer.InternetService != "No"
    if has_internet and customer.TechSupport == "No":
        factors.append("Sin soporte técnico contratado (41.6% tasa de churn)")

    if has_internet and customer.OnlineSecurity == "No":
        factors.append("Sin seguridad en línea contratada (41.8% tasa de churn)")

    if customer.PaymentMethod == "Electronic check":
        factors.append("Pago por cheque electrónico (45.3% tasa de churn histórica)")

    if customer.MonthlyCharges > 80:
        factors.append(f"Cargo mensual elevado (${customer.MonthlyCharges:.2f}/mes)")

    if customer.PaperlessBilling == "Yes":
        factors.append("Facturación electrónica habilitada (33.6% tasa de churn)")

    return factors[:3]


def business_recommendation(risk_level: str) -> str:
    recommendations = {
        "Alto": (
            "Cliente con alto riesgo de abandono. Se recomienda priorizar una acción "
            "de retención: ofrecer descuento personalizado, cambio a contrato anual "
            "o contacto directo del equipo comercial en los próximos 7 días."
        ),
        "Medio": (
            "Cliente con riesgo moderado. Se recomienda monitorear su comportamiento "
            "y evaluar incentivos preventivos como servicios adicionales gratuitos, "
            "mejoras de plan o campaña de fidelización."
        ),
        "Bajo": (
            "Cliente con bajo riesgo de abandono. Se recomienda mantener la relación "
            "actual, reforzar la satisfacción del servicio y fidelizar con beneficios "
            "por lealtad a largo plazo."
        )
    }
    return recommendations[risk_level]


def predict_customer(customer: CustomerData, threshold: float = 0.50) -> PredictionResult:
    input_df = customer_to_dataframe(customer)

    churn_probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(churn_probability >= threshold)

    prediction_label = "Churn" if prediction == 1 else "No Churn"
    risk_level = classify_risk(churn_probability)
    confidence = classify_confidence(churn_probability)
    risk_factors = get_risk_factors(customer, churn_probability)
    recommendation = business_recommendation(risk_level)

    logger.info(
        f"Predicción | prob={churn_probability:.4f} label={prediction_label} "
        f"risk={risk_level} confidence={confidence}"
    )

    return PredictionResult(
        prediction=prediction,
        prediction_label=prediction_label,
        churn_probability=round(churn_probability, 4),
        confidence=confidence,
        risk_level=risk_level,
        risk_factors=risk_factors,
        recommendation=recommendation,
        threshold_used=threshold
    )

# ============================================================
# Generador de la página HTML
# ============================================================

def _build_html() -> str:
    model_name = metadata.get("best_model_name", "XGBoost")
    test_auc = float(best_model_metrics.get("test_roc_auc", 0.8481))
    test_acc = float(best_model_metrics.get("test_accuracy", 0.8041))
    test_f1 = float(best_model_metrics.get("test_f1_score", 0.5831))

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Telco Churn Prediction</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:linear-gradient(135deg,#f0f4f8,#e8edf5);color:#1e293b;min-height:100vh}}
.hero{{background:linear-gradient(135deg,#0f172a,#1a56db);color:white;padding:56px 6%;text-align:center}}
.hero-svg{{margin-bottom:12px}}
.hero h1{{font-size:clamp(22px,4vw,40px);font-weight:800;margin-bottom:10px}}
.hero p{{font-size:15px;opacity:.88;max-width:680px;margin:0 auto 22px;line-height:1.65}}
.btns{{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}}
.btn{{padding:9px 18px;border-radius:8px;font-weight:700;font-size:13px;cursor:pointer;text-decoration:none;border:none;transition:all .2s}}
.btn-p{{background:#1a56db;color:white}}
.btn-s{{background:rgba(255,255,255,.15);color:white;border:1px solid rgba(255,255,255,.3)}}
.btn:hover{{opacity:.85;transform:translateY(-1px)}}
.mbar{{background:white;padding:18px 6%;display:flex;justify-content:center;gap:44px;flex-wrap:wrap;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
.mi{{text-align:center}}
.mv{{font-size:22px;font-weight:800;color:#1a56db}}
.ml{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}}
.wrap{{max-width:1080px;margin:0 auto;padding:36px 6%}}
.sec{{font-size:20px;font-weight:800;color:#0f172a;margin-bottom:16px;padding-bottom:8px;border-bottom:3px solid #1a56db}}
.card{{background:white;border-radius:16px;padding:28px;box-shadow:0 4px 20px rgba(15,23,42,.08);margin-bottom:20px}}
.ft{{font-size:13px;font-weight:700;color:#1a56db;text-transform:uppercase;letter-spacing:1px;margin:22px 0 14px}}
.ft:first-child{{margin-top:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}}
.fg{{display:flex;flex-direction:column;gap:5px}}
label{{font-size:12px;font-weight:700;color:#374151}}
select,input[type=number]{{padding:9px 11px;border:1.5px solid #e2e8f0;border-radius:8px;font-size:13px;color:#1e293b;background:white;width:100%;transition:border-color .2s}}
select:focus,input:focus{{outline:none;border-color:#1a56db;box-shadow:0 0 0 3px rgba(26,86,219,.1)}}
.pbtn{{width:100%;padding:13px;margin-top:20px;background:linear-gradient(135deg,#1a56db,#0f172a);color:white;font-size:15px;font-weight:800;border:none;border-radius:10px;cursor:pointer;transition:all .2s;letter-spacing:.5px}}
.pbtn:hover{{opacity:.9;transform:translateY(-1px);box-shadow:0 4px 14px rgba(26,86,219,.35)}}
.pbtn:disabled{{opacity:.6;cursor:not-allowed;transform:none}}
#rs{{display:none}}
.rc{{border-radius:16px;padding:28px;box-shadow:0 4px 20px rgba(0,0,0,.1);animation:fi .4s ease}}
@keyframes fi{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
.rc.ch{{background:linear-gradient(135deg,#fff5f5,#ffe4e4);border-left:5px solid #e02424}}
.rc.nc{{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-left:5px solid #057a55}}
.rh{{display:flex;align-items:center;gap:14px;margin-bottom:20px}}
.ri{{font-size:38px}}
.rl{{font-size:26px;font-weight:900}}
.rl.ch-l{{color:#e02424}}
.rl.nc-l{{color:#057a55}}
.rsub{{font-size:13px;color:#64748b;margin-top:2px}}
.rg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:14px;margin-bottom:16px}}
.rs{{background:white;border-radius:10px;padding:14px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.rv{{font-size:20px;font-weight:900}}
.rsl{{font-size:10px;color:#64748b;text-transform:uppercase;margin-top:3px}}
.rfs{{margin-top:14px}}
.rfs h4{{font-size:13px;font-weight:800;color:#374151;margin-bottom:8px}}
.rfi{{background:white;border-radius:8px;padding:9px 13px;margin-bottom:5px;font-size:13px;color:#374151;border-left:3px solid #f59e0b;display:flex;align-items:center;gap:8px}}
.rec{{background:white;border-radius:10px;padding:14px;margin-top:14px;font-size:13px;line-height:1.65;color:#374151;border-left:4px solid #1a56db}}
.rec strong{{display:block;margin-bottom:4px;color:#1a56db;font-size:12px;text-transform:uppercase;letter-spacing:.5px}}
.ic{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-top:18px}}
.ic-item{{background:white;border-radius:12px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.ic-item h4{{font-size:13px;font-weight:800;color:#1a56db;margin-bottom:6px}}
.ic-item p{{font-size:12px;color:#64748b;line-height:1.55}}
.spin{{width:18px;height:18px;border:3px solid rgba(255,255,255,.3);border-top-color:white;border-radius:50%;animation:sp .8s linear infinite;display:inline-block;margin-right:8px;vertical-align:middle}}
@keyframes sp{{to{{transform:rotate(360deg)}}}}
@media(max-width:600px){{.mbar{{gap:18px}}.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<section class="hero">
<svg class="hero-svg" width="56" height="56" viewBox="0 0 56 56" fill="none">
  <circle cx="28" cy="28" r="28" fill="rgba(255,255,255,0.15)"/>
  <path d="M20 36 L28 20 L36 36" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="28" cy="38" r="2" fill="white"/>
  <path d="M14 28 Q28 8 42 28" stroke="rgba(255,255,255,0.5)" stroke-width="2" fill="none" stroke-linecap="round"/>
  <path d="M10 32 Q28 4 46 32" stroke="rgba(255,255,255,0.25)" stroke-width="2" fill="none" stroke-linecap="round"/>
</svg>
<h1>Telco Customer Churn Prediction</h1>
<p>Sistema de predicción de abandono de clientes basado en <strong>Machine Learning</strong>.
Ingresa los datos del cliente para estimar su probabilidad de churn con el modelo <strong>{model_name}</strong>.</p>
<div class="btns">
  <a href="/docs" class="btn btn-p">📚 Documentación API</a>
  <a href="/health" class="btn btn-s">✅ Estado del Servicio</a>
  <a href="/model-info" class="btn btn-s">📊 Métricas del Modelo</a>
</div>
</section>

<div class="mbar">
  <div class="mi"><div class="mv">{model_name}</div><div class="ml">Mejor Modelo</div></div>
  <div class="mi"><div class="mv">{test_auc:.4f}</div><div class="ml">AUC-ROC (Test)</div></div>
  <div class="mi"><div class="mv">{test_acc:.1%}</div><div class="ml">Accuracy (Test)</div></div>
  <div class="mi"><div class="mv">{test_f1:.4f}</div><div class="ml">F1-Score (Test)</div></div>
  <div class="mi"><div class="mv">7,043</div><div class="ml">Clientes en Dataset</div></div>
</div>

<div class="wrap">
<h2 class="sec">Predicción de Churn</h2>
<div class="card">
  <div class="ft">👤 Datos Demográficos</div>
  <div class="grid">
    <div class="fg"><label>Género</label>
      <select id="gender"><option value="Female">Femenino</option><option value="Male">Masculino</option></select></div>
    <div class="fg"><label>Adulto Mayor</label>
      <select id="SeniorCitizen"><option value="0">No</option><option value="1">Sí</option></select></div>
    <div class="fg"><label>Tiene Pareja</label>
      <select id="Partner"><option value="Yes">Sí</option><option value="No">No</option></select></div>
    <div class="fg"><label>Tiene Dependientes</label>
      <select id="Dependents"><option value="No">No</option><option value="Yes">Sí</option></select></div>
    <div class="fg"><label>Antigüedad (meses)</label>
      <input type="number" id="tenure" value="12" min="0" max="72"></div>
  </div>

  <div class="ft">📱 Servicios Contratados</div>
  <div class="grid">
    <div class="fg"><label>Servicio Telefónico</label>
      <select id="PhoneService"><option value="Yes">Sí</option><option value="No">No</option></select></div>
    <div class="fg"><label>Múltiples Líneas</label>
      <select id="MultipleLines"><option value="No">No</option><option value="Yes">Sí</option><option value="No phone service">Sin servicio</option></select></div>
    <div class="fg"><label>Servicio de Internet</label>
      <select id="InternetService" onchange="syncInternet()"><option value="Fiber optic">Fibra óptica</option><option value="DSL">DSL</option><option value="No">Sin internet</option></select></div>
    <div class="fg"><label>Seguridad en Línea</label>
      <select id="OnlineSecurity"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin internet</option></select></div>
    <div class="fg"><label>Respaldo en Línea</label>
      <select id="OnlineBackup"><option value="Yes">Sí</option><option value="No">No</option><option value="No internet service">Sin internet</option></select></div>
    <div class="fg"><label>Protección Dispositivos</label>
      <select id="DeviceProtection"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin internet</option></select></div>
    <div class="fg"><label>Soporte Técnico</label>
      <select id="TechSupport"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin internet</option></select></div>
    <div class="fg"><label>Streaming TV</label>
      <select id="StreamingTV"><option value="Yes">Sí</option><option value="No">No</option><option value="No internet service">Sin internet</option></select></div>
    <div class="fg"><label>Streaming Películas</label>
      <select id="StreamingMovies"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin internet</option></select></div>
  </div>

  <div class="ft">📋 Contrato y Facturación</div>
  <div class="grid">
    <div class="fg"><label>Tipo de Contrato</label>
      <select id="Contract"><option value="Month-to-month">Mes a mes</option><option value="One year">Un año</option><option value="Two year">Dos años</option></select></div>
    <div class="fg"><label>Facturación Electrónica</label>
      <select id="PaperlessBilling"><option value="Yes">Sí</option><option value="No">No</option></select></div>
    <div class="fg"><label>Método de Pago</label>
      <select id="PaymentMethod">
        <option value="Electronic check">Cheque electrónico</option>
        <option value="Mailed check">Cheque por correo</option>
        <option value="Bank transfer (automatic)">Débito automático</option>
        <option value="Credit card (automatic)">Tarjeta de crédito</option>
      </select></div>
    <div class="fg"><label>Cargo Mensual ($)</label>
      <input type="number" id="MonthlyCharges" value="70.35" min="0" step="0.01"></div>
    <div class="fg"><label>Cargo Total ($)</label>
      <input type="number" id="TotalCharges" value="845.50" min="0" step="0.01"></div>
  </div>

  <button class="pbtn" id="pbtn" onclick="predict()">🔍 Analizar Riesgo de Churn</button>
</div>

<div id="rs">
<h2 class="sec">Resultado del Análisis</h2>
<div id="rc"></div>
</div>

<h2 class="sec" style="margin-top:36px">Acerca del Modelo</h2>
<div class="ic">
  <div class="ic-item"><h4>🤖 Algoritmo</h4>
    <p>{model_name} con GridSearchCV (5-fold StratifiedKFold, scoring=roc_auc).</p></div>
  <div class="ic-item"><h4>📊 Dataset</h4>
    <p>IBM Telco Customer Churn: 7,043 clientes, 19 features, desbalance 73%/27%.</p></div>
  <div class="ic-item"><h4>🎯 Objetivo</h4>
    <p>Predecir si un cliente abandonará la empresa en el próximo período de facturación.</p></div>
  <div class="ic-item"><h4>⚙️ Pipeline</h4>
    <p>SimpleImputer → StandardScaler (num) + OneHotEncoder (cat) → Clasificador.</p></div>
</div>
</div>

<script>
function val(id){{return document.getElementById(id).value}}
function ival(id){{return parseInt(document.getElementById(id).value)}}
function fval(id){{return parseFloat(document.getElementById(id).value)}}

function syncInternet(){{
  const net=val('InternetService');
  const fields=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'];
  fields.forEach(f=>{{
    const s=document.getElementById(f);
    if(net==='No'){{s.value='No internet service'}}
    else if(s.value==='No internet service'){{s.value='No'}}
  }});
}}

function riskColor(r){{return{{Alto:'#e02424',Medio:'#f59e0b',Bajo:'#057a55'}}[r]||'#64748b'}}
function confIcon(c){{return{{Alta:'🎯',Media:'📊',Baja:'❓'}}[c]||'📊'}}

async function predict(){{
  const btn=document.getElementById('pbtn');
  const rs=document.getElementById('rs');
  const rc=document.getElementById('rc');
  btn.disabled=true;
  btn.innerHTML='<span class="spin"></span>Analizando...';
  try{{
    const p={{
      gender:val('gender'),SeniorCitizen:ival('SeniorCitizen'),
      Partner:val('Partner'),Dependents:val('Dependents'),tenure:ival('tenure'),
      PhoneService:val('PhoneService'),MultipleLines:val('MultipleLines'),
      InternetService:val('InternetService'),OnlineSecurity:val('OnlineSecurity'),
      OnlineBackup:val('OnlineBackup'),DeviceProtection:val('DeviceProtection'),
      TechSupport:val('TechSupport'),StreamingTV:val('StreamingTV'),
      StreamingMovies:val('StreamingMovies'),Contract:val('Contract'),
      PaperlessBilling:val('PaperlessBilling'),PaymentMethod:val('PaymentMethod'),
      MonthlyCharges:fval('MonthlyCharges'),TotalCharges:fval('TotalCharges')
    }};
    const resp=await fetch('/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(p)}});
    if(!resp.ok){{const e=await resp.json();throw new Error(JSON.stringify(e.detail||'Error'));}}
    const d=await resp.json();const r=d.result;const isC=r.prediction===1;
    const rfHTML=r.risk_factors.length>0
      ?`<div class="rfs"><h4>⚠️ Factores de riesgo identificados</h4>${{r.risk_factors.map(f=>`<div class="rfi">⚡ ${{f}}</div>`).join('')}}</div>`
      :'';
    rc.innerHTML=`
    <div class="rc ${{isC?'ch':'nc'}}">
      <div class="rh">
        <div class="ri">${{isC?'⚠️':'✅'}}</div>
        <div><div class="rl ${{isC?'ch-l':'nc-l'}}">${{r.prediction_label}}</div>
        <div class="rsub">Análisis completado · ${{d.timestamp}}</div></div>
      </div>
      <div class="rg">
        <div class="rs"><div class="rv" style="color:${{riskColor(r.risk_level)}}">${{(r.churn_probability*100).toFixed(1)}}%</div><div class="rsl">Prob. Churn</div></div>
        <div class="rs"><div class="rv" style="color:${{riskColor(r.risk_level)}}">${{r.risk_level}}</div><div class="rsl">Nivel Riesgo</div></div>
        <div class="rs"><div class="rv">${{confIcon(r.confidence)}} ${{r.confidence}}</div><div class="rsl">Confianza</div></div>
        <div class="rs"><div class="rv">${{r.threshold_used}}</div><div class="rsl">Umbral</div></div>
      </div>
      ${{rfHTML}}
      <div class="rec"><strong>💼 Recomendación de negocio</strong>${{r.recommendation}}</div>
    </div>`;
    rs.style.display='block';
    rs.scrollIntoView({{behavior:'smooth',block:'start'}});
  }}catch(e){{
    rc.innerHTML=`<div class="rc ch"><div style="font-weight:700;margin-bottom:6px">❌ Error al procesar</div><div style="font-size:13px">${{e.message}}</div></div>`;
    rs.style.display='block';
  }}finally{{
    btn.disabled=false;
    btn.innerHTML='🔍 Analizar Riesgo de Churn';
  }}
}}
</script>
</body>
</html>"""

# ============================================================
# Endpoints
# ============================================================

@app.get("/", tags=["General"], response_class=HTMLResponse, include_in_schema=False)
def home():
    return _build_html()


@app.get("/health", tags=["General"], summary="Estado del servicio")
def health_check():
    return {
        "status": "ok",
        "message": "La API se encuentra activa.",
        "model_loaded": MODEL_PATH.exists(),
        "model_name": metadata.get("best_model_name", "Unknown"),
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get(
    "/model-info",
    tags=["Modelo"],
    summary="Información y métricas del modelo activo"
)
def model_info():
    metrics = {}
    if best_model_metrics:
        for key in ["test_accuracy", "test_precision", "test_recall",
                    "test_f1_score", "test_roc_auc", "best_cv_roc_auc"]:
            if key in best_model_metrics:
                metrics[key] = round(float(best_model_metrics[key]), 4)

    return {
        "model_name": metadata.get("best_model_name", "Unknown"),
        "model_status": "loaded",
        "version": "1.0.0",
        "selection_metric": metadata.get("selection_metric", "test_roc_auc"),
        "features_used": metadata.get("features_used", []),
        "numeric_features": metadata.get("numeric_features", []),
        "categorical_features": metadata.get("categorical_features", []),
        "prediction_threshold_default": 0.50,
        "risk_levels": {
            "Bajo": "probabilidad < 0.40",
            "Medio": "0.40 ≤ probabilidad < 0.70",
            "Alto": "probabilidad ≥ 0.70"
        },
        "confidence_levels": {
            "Alta": "|prob - 0.5| ≥ 0.25",
            "Media": "0.10 ≤ |prob - 0.5| < 0.25",
            "Baja": "|prob - 0.5| < 0.10"
        },
        "test_metrics": metrics
    }


@app.post(
    "/predict",
    tags=["Predicción"],
    response_model=PredictionResponse,
    summary="Predecir churn para un cliente",
    description=(
        "Recibe la información de un cliente y retorna la probabilidad estimada "
        "de abandono, el nivel de riesgo, factores de riesgo identificados y "
        "una recomendación de negocio."
    )
)
def predict(
    data: CustomerData,
    threshold: float = Query(
        0.50,
        ge=0.0,
        le=1.0,
        description="Umbral de decisión para clasificar al cliente como Churn."
    )
):
    result = predict_customer(data, threshold)
    return PredictionResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_data=data,
        result=result
    )


@app.post(
    "/predict-batch",
    tags=["Predicción"],
    response_model=BatchPredictionResponse,
    summary="Predecir churn para múltiples clientes",
    description="Recibe una lista de clientes y retorna la predicción para cada uno."
)
def predict_batch(
    data: BatchCustomerData,
    threshold: float = Query(
        0.50,
        ge=0.0,
        le=1.0,
        description="Umbral de decisión para clasificar clientes como Churn."
    )
):
    results = [predict_customer(customer, threshold) for customer in data.customers]
    return BatchPredictionResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_customers=len(data.customers),
        results=results
    )
