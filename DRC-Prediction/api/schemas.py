"""Pydantic schemas for the prediction API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    incidentType: str = Field(..., examples=["Hurricane"])
    declarationType: str = Field(..., examples=["DR"])
    state: str = Field(..., examples=["TX"])
    incident_duration_days: float = Field(..., ge=0, examples=[14])
    declaration_year: int = Field(..., ge=1953, le=2100, examples=[2023])
    rolling_5yr_freq: float = Field(..., ge=0, examples=[3.2])
    project_count: int = Field(..., ge=1, examples=[120])
    avg_project_amount: float = Field(..., ge=0, examples=[250000])
    is_high_cost_type: int = Field(..., ge=0, le=1, examples=[1])

    model_config = {"json_schema_extra": {
        "example": {
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
    }}


class PredictResponse(BaseModel):
    predicted_recovery_cost_usd: float
    model_name: str
    confidence_note: str
