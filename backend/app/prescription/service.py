"""
Prescription RAG Service.

Orchestrates prescription upload, extraction, indexing, and chat.
"""
import logging
from typing import Optional, List
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from app.prescription.models import Prescription, PrescriptionMedicine, PrescriptionChat
from app.prescription.schemas import (
    PrescriptionResponse,
    PrescriptionUploadResponse,
    MedicineResponse,
    ChatResponse,
    ChatHistoryResponse,
    ChatMessageBase,
)
from app.prescription.vision_ocr import get_vision_service
from app.prescription.rag_service import get_rag_service, get_llm_service
from app.prescription.langgraph_rag import get_langgraph_service

logger = logging.getLogger(__name__)


class PrescriptionService:
    """
    Service for managing prescriptions and RAG chat.
    """
    
    def __init__(self, db: AsyncSession):
        """Initialize with database session."""
        self.db = db
        self.vision_service = get_vision_service()
        self.rag_service = get_rag_service()
        self.llm_service = get_llm_service()
        self.langgraph_service = get_langgraph_service()  # LangGraph agent
    
    async def upload_and_process(
        self, 
        file_data: bytes, 
        filename: str,
        file_type: str,
        user_id: Optional[str] = None
    ) -> PrescriptionUploadResponse:
        """
        Upload and process a prescription file.
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            file_type: MIME type (image/jpeg, application/pdf, etc.)
            user_id: Optional user identifier for multi-user support
            
        Returns:
            PrescriptionUploadResponse with extracted medicines
        """
        # Create prescription record
        prescription = Prescription(
            filename=filename,
            file_type=file_type,
            file_size=len(file_data),
            status="processing",
            user_id=user_id
        )
        self.db.add(prescription)
        await self.db.commit()
        await self.db.refresh(prescription)
        
        try:
            # Extract medicines based on file type
            if file_type == "application/pdf":
                result = await self.vision_service.extract_from_pdf(file_data)
            else:
                result = await self.vision_service.extract_from_image(file_data, filename)
            
            # Update prescription with extraction results
            prescription.raw_text = result.raw_text
            prescription.extraction_confidence = result.confidence
            prescription.vision_model_used = result.model_used
            prescription.processed_at = datetime.now(timezone.utc)
            
            if result.error and not result.medicines:
                prescription.status = "failed"
                prescription.error_message = result.error
                await self.db.commit()
                
                return PrescriptionUploadResponse(
                    id=prescription.id,
                    status="failed",
                    message=f"Extraction failed: {result.error}",
                    medicines=[],
                    raw_text=result.raw_text,
                    extraction_confidence=result.confidence,
                    vision_model_used=result.model_used
                )
            
            # Save extracted medicines
            medicine_responses = []
            for med_data in result.medicines:
                medicine = PrescriptionMedicine(
                    prescription_id=prescription.id,
                    name=med_data.name,
                    generic_name=med_data.generic_name,
                    quantity=med_data.quantity,
                    dosage=med_data.dosage,
                    frequency=med_data.frequency,
                    duration=med_data.duration,
                    instructions=med_data.instructions,
                    morning=med_data.morning,
                    afternoon=med_data.afternoon,
                    evening=med_data.evening,
                    night=med_data.night,
                    extraction_confidence=result.confidence
                )
                self.db.add(medicine)
                await self.db.flush()
                
                medicine_responses.append(MedicineResponse(
                    id=medicine.id,
                    prescription_id=prescription.id,
                    name=medicine.name,
                    generic_name=medicine.generic_name,
                    quantity=medicine.quantity,
                    dosage=medicine.dosage,
                    frequency=medicine.frequency,
                    duration=medicine.duration,
                    instructions=medicine.instructions,
                    morning=medicine.morning,
                    afternoon=medicine.afternoon,
                    evening=medicine.evening,
                    night=medicine.night,
                    extraction_confidence=medicine.extraction_confidence,
                    created_at=medicine.created_at
                ))
            
            # Index in ChromaDB for RAG
            collection_id = self.rag_service.index_prescription(
                prescription_id=prescription.id,
                raw_text=result.raw_text,
                medicines=result.medicines
            )
            prescription.chroma_collection_id = collection_id
            
            prescription.status = "completed"
            await self.db.commit()
            
            return PrescriptionUploadResponse(
                id=prescription.id,
                status="completed",
                message=f"Successfully extracted {len(medicine_responses)} medicine(s)",
                medicines=medicine_responses,
                raw_text=result.raw_text[:1000] if result.raw_text else None,
                extraction_confidence=result.confidence,
                vision_model_used=result.model_used
            )
            
        except Exception as e:
            logger.error(f"Error processing prescription: {e}", exc_info=True)
            prescription.status = "failed"
            prescription.error_message = str(e)
            await self.db.commit()
            
            return PrescriptionUploadResponse(
                id=prescription.id,
                status="failed",
                message=f"Processing error: {str(e)}",
                medicines=[]
            )
    
    async def get_prescription(self, prescription_id: int) -> Optional[PrescriptionResponse]:
        """Get a prescription by ID."""
        result = await self.db.execute(
            select(Prescription).where(Prescription.id == prescription_id)
        )
        prescription = result.scalar_one_or_none()
        
        if not prescription:
            return None
        
        # Load medicines
        medicines_result = await self.db.execute(
            select(PrescriptionMedicine).where(
                PrescriptionMedicine.prescription_id == prescription_id
            )
        )
        medicines = medicines_result.scalars().all()
        
        return PrescriptionResponse(
            id=prescription.id,
            filename=prescription.filename,
            file_type=prescription.file_type,
            status=prescription.status,
            extraction_confidence=prescription.extraction_confidence,
            vision_model_used=prescription.vision_model_used,
            created_at=prescription.created_at,
            processed_at=prescription.processed_at,
            medicines=[MedicineResponse.model_validate(m) for m in medicines],
            error_message=prescription.error_message
        )
    
    async def list_prescriptions(
        self, 
        limit: int = 20, 
        offset: int = 0,
        user_id: Optional[str] = None
    ) -> tuple[List[PrescriptionResponse], int]:
        """List prescriptions with pagination, optionally filtered by user."""
        # Build base filter
        filters = []
        if user_id:
            filters.append(Prescription.user_id == user_id)
        
        # Get total count via SQL COUNT (no rows loaded into memory)
        count_q = select(func.count(Prescription.id))
        if filters:
            count_q = count_q.where(*filters)
        count_result = await self.db.execute(count_q)
        total = count_result.scalar() or 0
        
        base_query = select(Prescription)
        if filters:
            base_query = base_query.where(*filters)
        
        # Get prescriptions
        result = await self.db.execute(
            base_query
            .order_by(desc(Prescription.created_at))
            .offset(offset)
            .limit(limit)
        )
        prescriptions = result.scalars().all()
        
        # Build responses
        responses = []
        for p in prescriptions:
            # Load medicines for each prescription
            medicines_result = await self.db.execute(
                select(PrescriptionMedicine).where(
                    PrescriptionMedicine.prescription_id == p.id
                )
            )
            medicines = medicines_result.scalars().all()
            
            responses.append(PrescriptionResponse(
                id=p.id,
                filename=p.filename,
                file_type=p.file_type,
                status=p.status,
                extraction_confidence=p.extraction_confidence,
                vision_model_used=p.vision_model_used,
                created_at=p.created_at,
                processed_at=p.processed_at,
                medicines=[MedicineResponse.model_validate(m) for m in medicines],
                error_message=p.error_message
            ))
        
        return responses, total
    
    async def delete_prescription(self, prescription_id: int) -> bool:
        """Delete a prescription and its related data."""
        result = await self.db.execute(
            select(Prescription).where(Prescription.id == prescription_id)
        )
        prescription = result.scalar_one_or_none()
        
        if not prescription:
            return False
        
        # Delete from ChromaDB
        self.rag_service.delete_prescription(prescription_id)
        
        # Delete from database (cascade deletes medicines and chats)
        await self.db.delete(prescription)
        await self.db.commit()
        
        return True
    
    async def chat(
        self, 
        prescription_id: int, 
        message: str
    ) -> Optional[ChatResponse]:
        """
        Chat about a prescription.
        
        Args:
            prescription_id: ID of the prescription
            message: User's question
            
        Returns:
            ChatResponse with assistant's answer
        """
        # Verify prescription exists
        result = await self.db.execute(
            select(Prescription).where(Prescription.id == prescription_id)
        )
        prescription = result.scalar_one_or_none()
        
        if not prescription:
            return None
        
        # Get chat history
        history_result = await self.db.execute(
            select(PrescriptionChat)
            .where(PrescriptionChat.prescription_id == prescription_id)
            .order_by(PrescriptionChat.created_at)
        )
        chat_history = history_result.scalars().all()
        
        # Build history for context
        history_list = [
            {"role": msg.role, "content": msg.content}
            for msg in chat_history[-10:]  # Last 10 messages
        ]
        
        # Use LangGraph agent for sophisticated routing and tool calling
        # The agent handles: retrieval, drug lookup, interaction check, and response generation
        answer, model_used = await self.langgraph_service.answer_question(
            question=message,
            prescription_id=prescription_id,
            chat_history=history_list
        )
        
        # Get context for logging (separate query)
        context = self.rag_service.query(prescription_id, message)
        
        # Save user message
        user_chat = PrescriptionChat(
            prescription_id=prescription_id,
            role="user",
            content=message
        )
        self.db.add(user_chat)
        
        # Save assistant message
        assistant_chat = PrescriptionChat(
            prescription_id=prescription_id,
            role="assistant",
            content=answer,
            retrieved_context=context[:1000] if context else None,
            model_used=model_used
        )
        self.db.add(assistant_chat)
        
        await self.db.commit()
        
        return ChatResponse(
            prescription_id=prescription_id,
            user_message=message,
            assistant_message=answer,
            model_used=model_used,
            retrieved_context=context[:500] if context else None
        )
    
    async def get_chat_history(
        self, 
        prescription_id: int
    ) -> Optional[ChatHistoryResponse]:
        """Get chat history for a prescription."""
        # Verify prescription exists
        result = await self.db.execute(
            select(Prescription).where(Prescription.id == prescription_id)
        )
        prescription = result.scalar_one_or_none()
        
        if not prescription:
            return None
        
        # Get chat messages
        history_result = await self.db.execute(
            select(PrescriptionChat)
            .where(PrescriptionChat.prescription_id == prescription_id)
            .order_by(PrescriptionChat.created_at)
        )
        messages = history_result.scalars().all()
        
        return ChatHistoryResponse(
            prescription_id=prescription_id,
            messages=[
                ChatMessageBase(role=msg.role, content=msg.content)
                for msg in messages
            ]
        )
