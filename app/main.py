from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request

from app.model import MODEL_ID, MODEL_REVISION, load_classifier
from app.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = load_classifier()
    yield


app = FastAPI(title="Russian Text Toxicity API", lifespan=lifespan)


def make_prediction(toxicity: float) -> Prediction:
    if toxicity >= 0.5:
        return Prediction(label="toxic", probability=toxicity)
    return Prediction(label="non-toxic", probability=1 - toxicity)


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/predict")
def predict(payload: PredictRequest, request: Request) -> Prediction:
    (toxicity,) = request.app.state.classifier.predict([payload.text])
    return make_prediction(toxicity)


@app.post("/predict_batch")
def predict_batch(payload: BatchRequest, request: Request) -> BatchResponse:
    scores = request.app.state.classifier.predict(payload.texts)
    return BatchResponse(predictions=[make_prediction(score) for score in scores])


@app.get("/model_info")
def model_info() -> ModelInfo:
    return ModelInfo(
        name=MODEL_ID,
        revision=MODEL_REVISION,
        transformers_version=version("transformers"),
    )
