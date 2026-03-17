"""Top-level API router registration."""
from __future__ import annotations

from fastapi import APIRouter, FastAPI

from app.diabetic.router import router as diabetic_router
from app.prescription.router import router as prescription_router
from app.routers.admin import router as admin_router
from app.routers.adherence import router as adherence_router
from app.routers.drugs import router as drugs_router
from app.routers.history import router as history_router
from app.routers.interactions import router as interactions_router
from app.routers.ml_router import router as ml_router
from app.routers.ocr import router as ocr_router

API_ROUTERS = (
    diabetic_router,
    prescription_router,
    drugs_router,
    interactions_router,
    ocr_router,
    history_router,
    ml_router,
    adherence_router,
    admin_router,
)


def include_api_routers(app: FastAPI) -> None:
    """Register versioned and legacy routes."""
    v1_router = APIRouter(prefix="/v1")
    for router in API_ROUTERS:
        v1_router.include_router(router)
    app.include_router(v1_router)

    for router in API_ROUTERS:
        app.include_router(router)
