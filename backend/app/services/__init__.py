"""Services module for drug interaction system.

Imports are lazy to avoid hard failures when optional dependencies
(e.g., opencv-python for OCR) are not installed.
"""
from app.services.interaction_service import InteractionService, create_interaction_service
from app.services.data_fetcher import DrugDataFetcher
from app.services.comparison_logger import ComparisonLogger, create_comparison_logger

# Lazy import for OCR service — requires cv2 which may not be installed
try:
    from app.services.ocr_service import DrugOCRService, create_ocr_service
except ImportError:
    import logging
    logging.getLogger(__name__).warning(
        "OCR service unavailable — install opencv-python: pip install opencv-python"
    )
    DrugOCRService = None  # type: ignore[assignment,misc]

    def create_ocr_service(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "OCR service requires opencv-python. Install with: pip install opencv-python"
        )


__all__ = [
    "InteractionService",
    "create_interaction_service",
    "DrugOCRService",
    "create_ocr_service",
    "DrugDataFetcher",
    "ComparisonLogger",
    "create_comparison_logger",
]
