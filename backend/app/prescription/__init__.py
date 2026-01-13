"""
Prescription RAG Module.

Upload prescriptions, extract medicine details via Vision AI,
and chat about prescriptions using RAG.
"""
from app.prescription.router import router as prescription_router

__all__ = ["prescription_router"]
