from importlib.metadata import version

import pytest
from app.model import MODEL_ID, MODEL_REVISION
from fastapi.testclient import TestClient

from app import main


class StubClassifier:
    def predict(self, texts):
        scores = {"Доброе утро!": 0.125, "Ну ты и идиот!": 0.875}
        return [scores[text] for text in texts]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_classifier", StubClassifier)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_class_and_its_probability(client):
    response = client.post("/predict", json={"text": "Ну ты и идиот!"})

    assert response.status_code == 200
    assert response.json() == {"label": "toxic", "probability": 0.875}


def test_batch_preserves_input_order(client):
    response = client.post(
        "/predict_batch", json={"texts": ["Доброе утро!", "Ну ты и идиот!"]}
    )

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [
            {"label": "non-toxic", "probability": 0.875},
            {"label": "toxic", "probability": 0.875},
        ]
    }


def test_empty_batch_returns_empty_predictions(client):
    response = client.post("/predict_batch", json={"texts": []})

    assert response.status_code == 200
    assert response.json() == {"predictions": []}


def test_model_info_reports_model_revision_and_library_version(client):
    response = client.get("/model_info")

    assert response.status_code == 200
    assert response.json() == {
        "name": MODEL_ID,
        "revision": MODEL_REVISION,
        "transformers_version": version("transformers"),
    }
