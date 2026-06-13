from pydantic import BaseModel, Field


class IrisRequest(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class IrisResponse(BaseModel):
    prediction: str
    class_id: int
    confidence: float
    model_version: str
    request_id: str
    latency_ms: float