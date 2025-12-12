from fastapi import FastAPI
from api.routers.predict import router as predict_router

app = FastAPI(
    title="Risk Decision API",
    version="1.0.0",
    description="FastAPI service for model + rules + decision pipeline.",
)

app.include_router(predict_router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}
