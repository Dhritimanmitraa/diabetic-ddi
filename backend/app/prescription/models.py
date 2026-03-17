"""
Database models for Prescription RAG Module.

Stores prescription uploads, extracted medicines, and chat history.
"""
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database import Base


class Prescription(Base):
    """Uploaded prescription record."""
    __tablename__ = "prescriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # User ownership (for multi-user support)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    
    # File metadata
    filename = Column(String(255), nullable=True)
    file_type = Column(String(50), nullable=True)  # image/jpeg, image/png, application/pdf
    file_size = Column(Integer, nullable=True)  # bytes
    
    # Extracted content
    raw_text = Column(Text, nullable=True)  # Full OCR text
    extraction_confidence = Column(Float, nullable=True)  # 0-1
    
    # Processing status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Provider info
    vision_model_used = Column(String(100), nullable=True)  # gemini-1.5-flash, llava, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    processed_at = Column(DateTime, nullable=True)
    
    # ChromaDB reference
    chroma_collection_id = Column(String(255), nullable=True)
    
    # Relationships
    medicines = relationship("PrescriptionMedicine", back_populates="prescription", cascade="all, delete-orphan")
    chat_messages = relationship("PrescriptionChat", back_populates="prescription", cascade="all, delete-orphan")
    user = relationship("User", back_populates="prescriptions")
    
    def __repr__(self):
        return f"<Prescription(id={self.id}, status={self.status}, medicines={len(self.medicines)})>"


class PrescriptionMedicine(Base):
    """Extracted medicine from a prescription."""
    __tablename__ = "prescription_medicines"
    
    id = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=False)
    
    # Medicine details
    name = Column(String(255), nullable=False)
    generic_name = Column(String(255), nullable=True)
    quantity = Column(String(50), nullable=True)  # e.g., "10 tablets"
    dosage = Column(String(100), nullable=True)  # e.g., "500mg"
    frequency = Column(String(100), nullable=True)  # e.g., "1-0-1" or "twice daily"
    duration = Column(String(100), nullable=True)  # e.g., "7 days"
    instructions = Column(Text, nullable=True)  # Special instructions
    
    # Timing breakdown (parsed from frequency)
    morning = Column(Boolean, default=False)
    afternoon = Column(Boolean, default=False)
    evening = Column(Boolean, default=False)
    night = Column(Boolean, default=False)
    
    # Confidence
    extraction_confidence = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    prescription = relationship("Prescription", back_populates="medicines")
    
    def __repr__(self):
        return f"<PrescriptionMedicine(name={self.name}, dosage={self.dosage}, freq={self.frequency})>"


class PrescriptionChat(Base):
    """Chat messages for a prescription."""
    __tablename__ = "prescription_chats"
    
    id = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=False)
    
    # Message content
    role = Column(String(20), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    
    # RAG context (optional - for debugging)
    retrieved_context = Column(Text, nullable=True)
    
    # Model info
    model_used = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    prescription = relationship("Prescription", back_populates="chat_messages")
    
    def __repr__(self):
        return f"<PrescriptionChat(role={self.role}, content={self.content[:50]}...)>"
