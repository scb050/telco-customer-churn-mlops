# ============================================================
# Dockerfile - API de Predicción de Churn (Telco)
# Proyecto Integrador 5.12 - Capítulo 5 Machine Learning
# ============================================================

# Imagen base oficial de Python 3.10 en versión slim (sin herramientas de compilación)
# para mantener el contenedor lo más liviano posible
FROM python:3.10-slim

# Metadatos de la imagen para documentación y trazabilidad
LABEL maintainer="Proyecto ML - Telco Churn"
LABEL description="API FastAPI de predicción de churn para telecomunicaciones"
LABEL version="1.0.0"

# Variables de entorno:
# - MODEL_PATH: ruta al modelo serializado (configurable al iniciar el contenedor)
# - PYTHONDONTWRITEBYTECODE: evita generar archivos .pyc innecesarios
# - PYTHONUNBUFFERED: fuerza la salida estándar sin buffer (mejor para logs en Docker)
ENV MODEL_PATH=/app/models/best_model.joblib
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
# Todos los comandos siguientes se ejecutan relativos a este directorio
WORKDIR /app

# Copiar requirements.txt ANTES del código fuente
# Esto permite que Docker reutilice la capa de dependencias en builds posteriores
# siempre que requirements.txt no haya cambiado (optimización de caché de capas)
COPY requirements.txt .

# Instalar dependencias de Python sin caché para reducir el tamaño final de la imagen
# --no-cache-dir evita almacenar el caché de pip dentro del contenedor
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente de la aplicación FastAPI
COPY app/ ./app/

# Copiar los modelos entrenados y metadatos
COPY models/ ./models/

# Copiar los reportes de métricas del modelo (usados por /model-info)
COPY reports/ ./reports/

# Exponer el puerto 8000 donde la API atenderá solicitudes HTTP
# Nota: EXPOSE es documentacional; el puerto real se mapea con -p al correr el contenedor
EXPOSE 8000

# Health check: Docker verifica cada 30s que la API responda en /health
# --start-period=30s: espera 30s antes de la primera verificación (tiempo de carga del modelo)
# --retries=3: marca como unhealthy después de 3 fallos consecutivos
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando de inicio: uvicorn sirve la aplicación FastAPI
# --host 0.0.0.0: acepta conexiones desde cualquier interfaz de red del contenedor
# --port 8000: puerto expuesto
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
