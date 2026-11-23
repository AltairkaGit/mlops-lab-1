from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

InputText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class PredictRequest(BaseModel):
    text: InputText


class BatchRequest(BaseModel):
    texts: list[InputText]


class Prediction(BaseModel):
    label: Literal["non-toxic", "toxic"]
    probability: float = Field(ge=0, le=1)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    name: str
    revision: str
    transformers_version: str
