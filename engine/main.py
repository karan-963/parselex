"""FastAPI server for the Parselex resume inference engine."""
from __future__ import annotations

from fastapi import FastAPI

from inference_v2.routes import router as inference_v2_router

app = FastAPI(title="Parselex Inference Engine", version="1.0.0")
app.include_router(inference_v2_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
