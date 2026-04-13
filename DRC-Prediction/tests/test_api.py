import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def client():
    mock_artifact = {
        "pipeline": MagicMock(predict=MagicMock(return_value=np.array([15.5]))),
        "model_name": "XGBoost",
        "metrics": {"r2": 0.85, "rmse": 500000, "mae": 200000},
        "features": [],
        "target": "log_total_obligated",
    }
    with patch("api.main._model_artifact", mock_artifact):
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_cost(client):
    payload = {
        "incidentType": "Hurricane",
        "declarationType": "DR",
        "state": "FL",
        "incident_duration_days": 14,
        "declaration_year": 2023,
        "rolling_5yr_freq": 4.0,
        "project_count": 200,
        "avg_project_amount": 500000,
        "is_high_cost_type": 1,
    }
    resp = client.post("/predict-cost", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_recovery_cost_usd" in data
    assert data["predicted_recovery_cost_usd"] > 0


def test_predict(client):
    payload = {
        "incidentType": "Hurricane",
        "declarationType": "DR",
        "state": "FL",
        "incident_duration_days": -5,
        "declaration_year": 2023,
        "rolling_5yr_freq": 4.0,
        "project_count": 200,
        "avg_project_amount": 500000,
        "is_high_cost_type": 1,
    }
    resp = client.post("/predict-cost", json=payload)
    assert resp.status_code == 422



def test_predict_cost_missing_field(client):
    resp = client.post("/predict-cost", json={"incidentType": "Flood"})
    assert resp.status_code == 422
