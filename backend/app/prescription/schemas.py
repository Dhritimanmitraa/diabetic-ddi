"""
Pydantic schemas for Prescription RAG Module.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class PrescriptionStatus(str, Enum):
    """Prescription processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============== Medicine Schemas ==============

class MedicineBase(BaseModel):
    """Base medicine schema."""
    name: str
    generic_name: Optional[str] = None
    quantity: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instructions: Optional[str] = None
    morning: bool = False
    afternoon: bool = False
    evening: bool = False
    night: bool = False


class MedicineCreate(MedicineBase):
    """Create medicine schema."""
    pass


class MedicineResponse(MedicineBase):
    """Medicine response schema."""
    id: int
    prescription_id: int
    extraction_confidence: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============== Prescription Schemas ==============

class PrescriptionCreate(BaseModel):
    """Create prescription (internal use)."""
    filename: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None


class PrescriptionResponse(BaseModel):
    """Prescription response schema."""
    id: int
    filename: Optional[str] = None
    file_type: Optional[str] = None
    status: str
    extraction_confidence: Optional[float] = None
    vision_model_used: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    medicines: List[MedicineResponse] = []
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class PrescriptionUploadResponse(BaseModel):
    """Response after uploading a prescription."""
    id: int
    status: str
    message: str
    medicines: List[MedicineResponse] = []
    raw_text: Optional[str] = None
    extraction_confidence: Optional[float] = None
    vision_model_used: Optional[str] = None


# ============== Chat Schemas ==============

class ChatMessageBase(BaseModel):
    """Base chat message schema."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request schema."""
    prescription_id: int = Field(..., description="ID of the prescription to chat about")
    message: str = Field(..., min_length=1, description="User's question")


class ChatResponse(BaseModel):
    """Chat response schema."""
    prescription_id: int
    user_message: str
    assistant_message: str
    model_used: Optional[str] = None
    retrieved_context: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Chat history response schema."""
    prescription_id: int
    messages: List[ChatMessageBase]


# ============== History Schemas ==============

class PrescriptionHistoryResponse(BaseModel):
    """Prescription history list response."""
    total: int
    prescriptions: List[PrescriptionResponse]


# ============== Extraction Result (Internal) ==============

class ExtractionResult(BaseModel):
    """Result from vision OCR extraction."""
    raw_text: str
    medicines: List[MedicineCreate]
    confidence: float
    model_used: str
    error: Optional[str] = None
