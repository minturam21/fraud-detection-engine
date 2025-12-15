from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """
    Adjust 'features' to match your real input.
    For now it's a simple feature vector.
    """
    features: List[float] = Field(
        ...,
        description="Ordered feature vector for the model.",
        min_items=1
    )


class PredictionResponse(BaseModel):
    success: bool = Field(..., description="Whether the request was processed successfully.")
    prediction: str = Field(..., description="Final decision, e.g. ALLOW / OTP / BLOCK.")
    confidence: float = Field(..., description="Final risk score used for decision.")
    detail: Optional[list[str]] = Field(
        default=None,
        description="List of reason codes / explanations."
    )
