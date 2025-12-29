"""
Custom exception classes for the application.
"""
from fastapi import HTTPException, status
from typing import Optional


class DrugInteractionException(HTTPException):
    """Base exception for drug interaction errors."""
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "An error occurred",
        error_code: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


class DrugNotFoundError(DrugInteractionException):
    """Raised when a drug is not found."""
    
    def __init__(self, drug_name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drug '{drug_name}' not found",
            error_code="DRUG_NOT_FOUND"
        )


class ValidationError(DrugInteractionException):
    """Raised when input validation fails."""
    
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )


class MLModelError(DrugInteractionException):
    """Raised when ML model operations fail."""
    
    def __init__(self, detail: str = "ML model error occurred"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="ML_MODEL_ERROR"
        )


class RateLimitError(DrugInteractionException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_ERROR"
        )


class PatientNotFoundError(DrugInteractionException):
    """Raised when a patient is not found."""
    
    def __init__(self, patient_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found",
            error_code="PATIENT_NOT_FOUND"
        )


class MedicationNotFoundError(DrugInteractionException):
    """Raised when a medication is not found."""
    
    def __init__(self, medication_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Medication '{medication_id}' not found",
            error_code="MEDICATION_NOT_FOUND"
        )

