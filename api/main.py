"""
FastAPI — Servicio de Inferencia para Clasificación de Pingüinos
================================================================
Endpoints:
  GET  /              → Health check
  GET  /health        → Estado detallado del servicio
  GET  /model/info    → Info del modelo en producción
  POST /predict       → Predicción individual
  POST /predict/batch → Predicción por lote
  GET  /predictions   → Historial de predicciones (desde PostgreSQL)
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import create_engine, text
import mlflow
import mlflow.sklearn

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("penguins-api")

# ── Configuración ──────────────────────────────────────────────
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME   = "penguins-classifier"
MODEL_STAGE  = "Production"

DB_HOST = os.environ.get("DB_HOST", "postgres")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "mlops_db")
DB_USER = os.environ.get("DB_USER", "mlops")
DB_PASS = os.environ.get("DB_PASSWORD", "mlops_secret")
DB_URL  = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

LABEL_MAP = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

FEATURE_COLS = [
    "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
    "island_Dream", "island_Torgersen", "sex_male"
]

# ── Estado global del modelo ────────────────────────────────────
model_state = {
    "model": None,
    "run_id": None,
    "version": None,
    "loaded": False
}


def load_model():
    """Carga el modelo en producción desde MLflow Registry."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        uri   = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(uri)

        client  = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if versions:
            v = versions[0]
            model_state["run_id"]  = v.run_id
            model_state["version"] = v.version

        model_state["model"]  = model
        model_state["loaded"] = True
        logger.info(f"✅ Modelo cargado: {MODEL_NAME} v{model_state['version']} ({MODEL_STAGE})")
    except Exception as e:
        logger.error(f"❌ No se pudo cargar el modelo: {e}")
        model_state["loaded"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la aplicación."""
    load_model()
    yield


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="🐧 Penguins Classifier API",
    description="""
API de inferencia para clasificación de especies de pingüinos.

El modelo es cargado directamente desde **MLflow Model Registry** (stage: Production).

### Features de entrada
| Feature | Tipo | Descripción |
|---------|------|-------------|
| bill_length_mm | float | Longitud del pico (mm) |
| bill_depth_mm | float | Profundidad del pico (mm) |
| flipper_length_mm | float | Longitud de la aleta (mm) |
| body_mass_g | float | Masa corporal (g) |
| island | string | Isla: Biscoe, Dream, Torgersen |
| sex | string | Sexo: male, female |

### Clases predichas
- **Adelie** (0)
- **Chinstrap** (1)
- **Gentoo** (2)
    """,
    version="1.0.0",
    lifespan=lifespan
)

# ── Modelos Pydantic ────────────────────────────────────────────

class PenguinInput(BaseModel):
    bill_length_mm:    float = Field(..., gt=0, lt=100,  example=39.1,  description="Longitud del pico (mm)")
    bill_depth_mm:     float = Field(..., gt=0, lt=30,   example=18.7,  description="Profundidad del pico (mm)")
    flipper_length_mm: float = Field(..., gt=0, lt=300,  example=181.0, description="Longitud de la aleta (mm)")
    body_mass_g:       float = Field(..., gt=0, lt=7000, example=3750.0, description="Masa corporal (g)")
    island:            str   = Field(..., example="Torgersen", description="Isla: Biscoe | Dream | Torgersen")
    sex:               str   = Field(..., example="male",      description="Sexo: male | female")

    @field_validator("island")
    @classmethod
    def validate_island(cls, v):
        valid = {"Biscoe", "Dream", "Torgersen"}
        if v not in valid:
            raise ValueError(f"island debe ser uno de: {valid}")
        return v

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v):
        v = v.lower()
        if v not in {"male", "female"}:
            raise ValueError("sex debe ser 'male' o 'female'")
        return v

    def to_feature_vector(self) -> list:
        """Convierte la entrada al vector de features del modelo."""
        island_Dream     = 1 if self.island == "Dream"     else 0
        island_Torgersen = 1 if self.island == "Torgersen" else 0
        sex_male         = 1 if self.sex    == "male"      else 0
        return [
            self.bill_length_mm,
            self.bill_depth_mm,
            self.flipper_length_mm,
            self.body_mass_g,
            island_Dream,
            island_Torgersen,
            sex_male
        ]


class PredictionResponse(BaseModel):
    species:          str
    species_class:    int
    confidence:       float
    probabilities:    dict
    model_name:       str
    model_version:    Optional[str]
    run_id:           Optional[str]


class BatchPredictionRequest(BaseModel):
    penguins: list[PenguinInput]


# ── Helpers ────────────────────────────────────────────────────

def get_engine():
    return create_engine(DB_URL)


def log_prediction_to_db(
    input_data: PenguinInput,
    predicted_class: int,
    predicted_species: str,
    confidence: float
):
    """Registra la predicción en PostgreSQL."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO predictions_log (
                    run_id, model_version,
                    bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g,
                    island, sex,
                    predicted_species, predicted_class, confidence
                ) VALUES (
                    :run_id, :model_version,
                    :bill_length_mm, :bill_depth_mm, :flipper_length_mm, :body_mass_g,
                    :island, :sex,
                    :predicted_species, :predicted_class, :confidence
                )
            """), {
                "run_id":            model_state["run_id"],
                "model_version":     model_state["version"],
                "bill_length_mm":    input_data.bill_length_mm,
                "bill_depth_mm":     input_data.bill_depth_mm,
                "flipper_length_mm": input_data.flipper_length_mm,
                "body_mass_g":       input_data.body_mass_g,
                "island":            input_data.island,
                "sex":               input_data.sex,
                "predicted_species": predicted_species,
                "predicted_class":   predicted_class,
                "confidence":        confidence
            })
            conn.commit()
    except Exception as e:
        logger.warning(f"No se pudo registrar predicción en DB: {e}")


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Penguins Classifier API",
        "status":  "running",
        "model_loaded": model_state["loaded"],
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health():
    """Estado detallado: modelo, base de datos, MLflow."""
    # Verificar DB
    db_ok = False
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if model_state["loaded"] else "degraded",
        "model": {
            "loaded":  model_state["loaded"],
            "name":    MODEL_NAME,
            "version": model_state["version"],
            "stage":   MODEL_STAGE,
            "run_id":  model_state["run_id"]
        },
        "database": {"connected": db_ok},
        "mlflow_uri": MLFLOW_URI
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Información del modelo en producción desde MLflow."""
    if not model_state["loaded"]:
        raise HTTPException(503, "Modelo no disponible")
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            raise HTTPException(404, f"No hay versión en {MODEL_STAGE}")
        v = versions[0]
        run = client.get_run(v.run_id)
        return {
            "name":        MODEL_NAME,
            "version":     v.version,
            "stage":       MODEL_STAGE,
            "run_id":      v.run_id,
            "description": v.description,
            "tags":        v.tags,
            "metrics":     run.data.metrics,
            "params":      run.data.params,
            "features":    FEATURE_COLS,
            "classes":     LABEL_MAP
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(input_data: PenguinInput):
    """
    Realiza una predicción de especie de pingüino.

    Retorna la especie predicha, probabilidades por clase y metadatos del modelo.
    """
    if not model_state["loaded"]:
        raise HTTPException(503, "Modelo no disponible. Revisa /health")

    features = np.array([input_data.to_feature_vector()])

    try:
        pred_class = int(model_state["model"].predict(features)[0])
        probas     = model_state["model"].predict_proba(features)[0]
        confidence = float(probas.max())
        pred_label = LABEL_MAP[pred_class]

        probabilities = {
            LABEL_MAP[i]: round(float(p), 4)
            for i, p in enumerate(probas)
        }

        # Guardar en PostgreSQL
        log_prediction_to_db(input_data, pred_class, pred_label, confidence)

        return PredictionResponse(
            species=pred_label,
            species_class=pred_class,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model_name=MODEL_NAME,
            model_version=model_state["version"],
            run_id=model_state["run_id"]
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(500, f"Error de inferencia: {str(e)}")


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(request: BatchPredictionRequest):
    """
    Predicción por lote — acepta hasta 100 pingüinos por petición.
    """
    if not model_state["loaded"]:
        raise HTTPException(503, "Modelo no disponible")
    if len(request.penguins) > 100:
        raise HTTPException(400, "Máximo 100 predicciones por lote")

    results = []
    feature_matrix = np.array([p.to_feature_vector() for p in request.penguins])

    try:
        pred_classes = model_state["model"].predict(feature_matrix)
        pred_probas  = model_state["model"].predict_proba(feature_matrix)

        for i, (inp, pred_class, probas) in enumerate(
            zip(request.penguins, pred_classes, pred_probas)
        ):
            pred_class = int(pred_class)
            pred_label = LABEL_MAP[pred_class]
            confidence = float(probas.max())

            log_prediction_to_db(inp, pred_class, pred_label, confidence)

            results.append({
                "index": i,
                "species":       pred_label,
                "species_class": pred_class,
                "confidence":    round(confidence, 4),
                "probabilities": {
                    LABEL_MAP[j]: round(float(p), 4)
                    for j, p in enumerate(probas)
                }
            })

        return {
            "total": len(results),
            "model_version": model_state["version"],
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(500, f"Error en predicción batch: {str(e)}")


@app.get("/predictions", tags=["History"])
def get_predictions(limit: int = Query(default=20, le=200)):
    """Historial de predicciones almacenadas en PostgreSQL."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT id, predicted_species, predicted_class, confidence,
                       bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g,
                       island, sex, model_version, requested_at
                FROM predictions_log
                ORDER BY requested_at DESC
                LIMIT :limit
            """), {"limit": limit}).fetchall()

        return {
            "total": len(rows),
            "predictions": [dict(r._mapping) for r in rows]
        }
    except Exception as e:
        raise HTTPException(500, f"Error consultando predicciones: {str(e)}")


@app.post("/model/reload", tags=["Model"])
def reload_model():
    """Recarga el modelo desde MLflow Registry (sin reiniciar el servicio)."""
    load_model()
    if model_state["loaded"]:
        return {"status": "ok", "version": model_state["version"], "run_id": model_state["run_id"]}
    raise HTTPException(503, "No se pudo cargar el modelo")
