# api/routers/predict.py
from fastapi import APIRouter, HTTPException
from api.schemas.predict import PredictionRequest, PredictionResponse
from pipeline.model_pipeline import ModelPipeline
from scoring.decision_pipeline import DecisionPipeline
from utils.rules import compute_rule_score_and_flags
from utils.model_loader import ModelNotFoundError
from utils.postprocess import format_prediction_output

router = APIRouter(
    prefix="/v1",
    tags=["prediction"],
)

# Initialize pipelines once
try:
    model_pipeline = ModelPipeline()
except ModelNotFoundError as e:
    # If model is missing, fail fast on startup in real deployment
    # Here we keep it simple and raise at request time instead.
    model_pipeline = None

decision_pipeline = DecisionPipeline(
    threshold={
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
    }
)


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Main decision endpoint.
    1. Score with ML model
    2. Apply rule engine
    3. Combine scores
    4. Run decision pipeline
    5. Return final decision
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Check model file path / deployment.",
        )

    try:
        # Model score
        model_score = model_pipeline.score(request.features)

        #  Rule score + flags
        rule_score, rule_flags = compute_rule_score_and_flags(model_score)

        #  Combine into final_score (your business formula)
        final_score = (model_score + rule_score) / 2.0

        # Run decision pipeline
        decision = decision_pipeline.run(
            final_score=final_score,
            model_score=model_score,
            rule_score=rule_score,
            rule_flags=rule_flags,
            extra_context={"source": "api"},
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Decision computation failed: {str(e)}",
        )

    # Format response
    response_data = format_prediction_output(
        action=decision["action"],
        final_score=decision["final_score"],
        reasons=decision["reasons"],
    )

    return response_data
