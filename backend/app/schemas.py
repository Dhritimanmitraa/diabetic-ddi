"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field, field_validator, StringConstraints
from typing import Optional, List, Annotated
from datetime import datetime
from enum import Enum


class SeverityLevel(str, Enum):
    """Interaction severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"


class EvidenceLevel(str, Enum):
    """Evidence levels for interactions."""
    ESTABLISHED = "established"
    THEORETICAL = "theoretical"
    CASE_REPORT = "case_report"


# Drug Schemas
class DrugBase(BaseModel):
    """Base drug schema."""
    name: str
    generic_name: Optional[str] = None
    brand_names: Optional[str] = None
    description: Optional[str] = None
    drug_class: Optional[str] = None


class DrugCreate(DrugBase):
    """Schema for creating a drug."""
    drugbank_id: Optional[str] = None
    mechanism: Optional[str] = None
    indication: Optional[str] = None


class DrugResponse(DrugBase):
    """Schema for drug response."""
    id: int
    drugbank_id: Optional[str] = None
    is_approved: bool = True
    created_at: datetime
    
    class Config:
        from_attributes = True


class DrugSearch(BaseModel):
    """Schema for drug search."""
    query: str = Field(..., min_length=2, max_length=255)
    limit: int = Field(default=10, ge=1, le=100)


# Interaction Schemas
class InteractionBase(BaseModel):
    """Base interaction schema."""
    severity: SeverityLevel
    description: Optional[str] = None
    effect: Optional[str] = None
    mechanism: Optional[str] = None
    management: Optional[str] = None


class InteractionCreate(InteractionBase):
    """Schema for creating an interaction."""
    drug1_id: int
    drug2_id: int
    source: Optional[str] = None
    evidence_level: Optional[EvidenceLevel] = None
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)


class InteractionResponse(InteractionBase):
    """Schema for interaction response."""
    id: int
    drug1: DrugResponse
    drug2: DrugResponse
    source: Optional[str] = None
    evidence_level: Optional[str] = None
    confidence_score: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class InteractionCheckRequest(BaseModel):
    """Request to check interaction between two drugs."""
    drug1_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2)] = Field(..., description="First drug name")
    drug2_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2)] = Field(..., description="Second drug name")

    @field_validator("drug1_name", "drug2_name")
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.strip()

    @field_validator("drug2_name")
    @classmethod
    def not_same(cls, v: str, info):
        drug1 = info.data.get("drug1_name", "").strip().lower()
        if v.strip().lower() == drug1:
            raise ValueError("Drug names must be different")
        return v


class InteractionCheckResponse(BaseModel):
    """Response for interaction check."""
    drug1: DrugResponse
    drug2: DrugResponse
    has_interaction: bool
    is_safe: bool
    interaction: Optional[InteractionResponse] = None
    safety_message: str
    recommendations: List[str] = []
    # ML augmentation
    ml_probability: Optional[float] = None
    ml_severity: Optional[str] = None
    ml_decision_source: Optional[str] = None  # ml_primary | rule_override | rules_only


# Alternative Drug Schemas
class AlternativeDrug(BaseModel):
    """Alternative drug suggestion."""
    drug: DrugResponse
    similarity_score: float
    reason: str
    has_interaction_with_other: bool


class AlternativeSuggestionResponse(BaseModel):
    """Response with alternative drug suggestions."""
    original_drug1: DrugResponse
    original_drug2: DrugResponse
    alternatives_for_drug1: List[AlternativeDrug]
    alternatives_for_drug2: List[AlternativeDrug]
    safe_combinations: List[dict]


# OCR Schemas
class OCRRequest(BaseModel):
    """Request for OCR processing."""
    image_base64: str = Field(..., description="Base64 encoded image")


class OCRResponse(BaseModel):
    """Response from OCR processing."""
    extracted_text: str
    detected_drugs: List[str]
    confidence: float


# Statistics
class DatabaseStats(BaseModel):
    """Database statistics."""
    total_drugs: int
    total_interactions: int
    interactions_by_severity: dict
    last_updated: datetime

