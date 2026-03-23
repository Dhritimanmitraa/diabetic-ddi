"""
FastAPI Router for Prescription RAG Module.

Endpoints for uploading prescriptions, extracting medicines, and chatting.
Includes a WebSocket endpoint for real-time prescription Q&A.
"""
import json
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, async_session
from app.config import get_settings
from app.models import User
from app.prescription.service import PrescriptionService
from app.prescription.schemas import (
    PrescriptionResponse,
    PrescriptionUploadResponse,
    PrescriptionHistoryResponse,
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
)
from app.services.jwt_auth import decode_token, require_current_user
from app.services.rate_limiter import rate_limit_dependency


class DrugInteractionCheckRequest(BaseModel):
    """Request to check drug interactions."""
    drug_names: List[str]


class DrugInteractionItem(BaseModel):
    """Single drug interaction result."""
    drug1: str
    drug2: str
    severity: str
    description: Optional[str] = None


class DrugInteractionCheckResponse(BaseModel):
    """Response with drug interactions."""
    drugs_checked: List[str]
    interactions: List[DrugInteractionItem]
    total_interactions: int

logger = logging.getLogger(__name__)
settings = get_settings()
_health_cache: dict[str, object] = {"expires_at": 0.0, "payload": None}
_HEALTH_CACHE_TTL_SECONDS = 30.0

router = APIRouter(
    prefix="/prescription",
    tags=["Prescription RAG"],
    responses={404: {"description": "Not found"}},
)


# ============== Dependency ==============

def get_service(db: AsyncSession = Depends(get_db)) -> PrescriptionService:
    """Get prescription service instance."""
    return PrescriptionService(db)


# ============== Upload Endpoints ==============

@router.post("/upload", response_model=PrescriptionUploadResponse)
async def upload_prescription(
    file: UploadFile = File(..., description="Prescription image (JPEG, PNG) or PDF"),
    service: PrescriptionService = Depends(get_service),
    current_user: User = Depends(require_current_user),
    _: None = rate_limit_dependency(limit=settings.HEAVY_RATE_LIMIT_REQUESTS_PER_MIN, key_prefix="prescription_upload"),
):
    """
    Upload a prescription image or PDF for extraction.
    
    Supports:
    - Images: JPEG, PNG, WebP
    - Documents: PDF
    
    Returns extracted medicines with dosage, frequency, and timing information.
    """
    # Validate file type
    allowed_types = [
        "image/jpeg", "image/jpg", "image/png", "image/webp",
        "application/pdf"
    ]
    
    content_type = file.content_type or "application/octet-stream"
    
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Allowed: {', '.join(allowed_types)}"
        )
    
    # Read file
    try:
        file_data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")
    
    # Validate file size (max 10MB)
    if len(file_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    
    # Process prescription
    result = await service.upload_and_process(
        file_data=file_data,
        filename=file.filename or "prescription",
        file_type=content_type,
        user_id=current_user.id,
    )
    
    return result


@router.post("/upload/base64", response_model=PrescriptionUploadResponse)
async def upload_prescription_base64(
    image_base64: str = Form(..., description="Base64 encoded image"),
    filename: str = Form("prescription.jpg", description="Filename"),
    service: PrescriptionService = Depends(get_service),
    current_user: User = Depends(require_current_user),
    _: None = rate_limit_dependency(limit=settings.HEAVY_RATE_LIMIT_REQUESTS_PER_MIN, key_prefix="prescription_upload_b64"),
):
    """
    Upload a prescription as base64 encoded image.
    
    Useful for mobile apps and camera capture.
    """
    import base64
    
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        file_data = base64.b64decode(image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")

    if len(file_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    
    # Determine content type from filename
    ext = filename.lower().split('.')[-1] if '.' in filename else 'jpg'
    content_type_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'webp': 'image/webp',
        'pdf': 'application/pdf',
    }
    content_type = content_type_map.get(ext, 'image/jpeg')
    
    result = await service.upload_and_process(
        file_data=file_data,
        filename=filename,
        file_type=content_type,
        user_id=current_user.id,
    )
    
    return result


# ============== Prescription CRUD Endpoints ==============

@router.get("/history", response_model=PrescriptionHistoryResponse)
async def list_prescriptions(
    limit: int = 20,
    offset: int = 0,
    service: PrescriptionService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    List uploaded prescriptions with pagination.
    
    Optionally filter by user_id for multi-user support.
    """
    prescriptions, total = await service.list_prescriptions(current_user.id, limit, offset)
    
    return PrescriptionHistoryResponse(
        total=total,
        prescriptions=prescriptions
    )


@router.get("/{prescription_id}", response_model=PrescriptionResponse)
async def get_prescription(
    prescription_id: int,
    service: PrescriptionService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Get a specific prescription by ID.
    """
    prescription = await service.get_prescription(prescription_id, current_user.id)
    
    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    return prescription


@router.delete("/{prescription_id}")
async def delete_prescription(
    prescription_id: int,
    service: PrescriptionService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Delete a prescription and all related data.
    """
    success = await service.delete_prescription(prescription_id, current_user.id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    return {"message": "Prescription deleted successfully", "id": prescription_id}


# ============== Chat Endpoints ==============

@router.post("/chat", response_model=ChatResponse)
async def chat_with_prescription(
    request: ChatRequest,
    service: PrescriptionService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Ask a question about a prescription.
    
    Uses RAG to retrieve relevant context and LLM to generate answers.
    
    Example questions:
    - "When should I take Nucoxia-MR?"
    - "What is the dosage for PAN 40?"
    - "How many tablets should I take in the morning?"
    """
    result = await service.chat(
        prescription_id=request.prescription_id,
        message=request.message,
        user_id=current_user.id,
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    return result


@router.get("/{prescription_id}/chat-history", response_model=ChatHistoryResponse)
async def get_chat_history(
    prescription_id: int,
    service: PrescriptionService = Depends(get_service),
    current_user: User = Depends(require_current_user),
):
    """
    Get chat history for a prescription.
    """
    history = await service.get_chat_history(prescription_id, current_user.id)
    
    if not history:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    return history


# ============== Drug Interaction Check ==============

@router.post("/check-interactions", response_model=DrugInteractionCheckResponse)
async def check_drug_interactions(
    request: DrugInteractionCheckRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Check for drug interactions between prescription medicines.
    
    Takes a list of drug names and checks all pairwise interactions.
    """
    from app.services.interaction_service import InteractionService
    
    drug_names = request.drug_names
    if len(drug_names) < 2:
        return DrugInteractionCheckResponse(
            drugs_checked=drug_names,
            interactions=[],
            total_interactions=0
        )
    
    service = InteractionService(db)
    interactions = []

    for drug1, drug2, result in await service.check_batch_interactions(drug_names):
        if result.has_interaction and result.interaction:
            interactions.append(
                DrugInteractionItem(
                    drug1=drug1,
                    drug2=drug2,
                    severity=result.interaction.severity.value if hasattr(result.interaction.severity, "value") else str(result.interaction.severity),
                    description=result.interaction.description,
                )
            )

    # Sort by severity (contraindicated > major > moderate > minor)
    severity_order = {'contraindicated': 0, 'major': 1, 'moderate': 2, 'minor': 3}
    interactions.sort(key=lambda x: severity_order.get(x.severity.lower(), 4))
    
    return DrugInteractionCheckResponse(
        drugs_checked=drug_names,
        interactions=interactions,
        total_interactions=len(interactions)
    )


# ============== Health Check ==============

@router.get("/health/status")
async def prescription_health():
    """
    Check prescription module health.
    """
    from app.prescription.vision_ocr import get_vision_service
    from app.prescription.rag_service import get_rag_service, get_llm_service
    
    now = time.monotonic()
    cached_payload = _health_cache.get("payload")
    expires_at = float(_health_cache.get("expires_at", 0.0))
    if cached_payload is not None and now < expires_at:
        return cached_payload

    vision = get_vision_service()
    rag = get_rag_service()
    llm = get_llm_service()

    payload = {
        "status": "healthy",
        "services": {
            "nvidia_cosmos": vision.nvidia_available,
            "gemini_vision": vision.gemini_available,
            "ollama_fallback": bool(settings.OLLAMA_HOST and settings.OLLAMA_MODEL),
            "chromadb": rag.chroma_client is not None,
            "gemini_chat": llm.gemini_available,
        },
    }
    _health_cache["payload"] = payload
    _health_cache["expires_at"] = now + _HEALTH_CACHE_TTL_SECONDS
    return payload


# ============== WebSocket Chat ==============

@router.websocket("/ws/chat/{prescription_id}")
async def websocket_chat(websocket: WebSocket, prescription_id: int):
    """
    Real-time prescription Q&A over WebSocket.

    Protocol (JSON messages):
      Client → Server: {"message": "When should I take PAN 40?"}
      Server → Client: {"type": "answer", "message": "...", "sources": [...]}
      Server → Client: {"type": "error", "message": "..."}
    """
    token = (websocket.query_params.get("token") or "").strip()
    if not token:
        await websocket.close(code=1008, reason="Missing access token")
        return

    try:
        payload = decode_token(token, expected_type="access")
        current_user_id = payload["sub"]
    except HTTPException:
        await websocket.close(code=1008, reason="Invalid access token")
        return

    await websocket.accept()
    logger.info(
        "WebSocket connected",
        extra={"prescription_id": prescription_id, "user_id": current_user_id},
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            user_msg = data.get("message", "").strip()
            if not user_msg:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue

            # Use a fresh DB session per message to avoid stale-object issues.
            async with async_session() as session:
                service = PrescriptionService(session)
                result = await service.chat(
                    prescription_id=prescription_id,
                    message=user_msg,
                    user_id=current_user_id,
                )

            if result is None:
                await websocket.send_json(
                    {"type": "error", "message": "Prescription not found"}
                )
                continue

            await websocket.send_json({
                "type": "answer",
                "message": result.assistant_message,
                "model_used": result.model_used or "",
                "context": result.retrieved_context or "",
            })

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected",
            extra={"prescription_id": prescription_id, "user_id": current_user_id},
        )
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Unable to process the request right now. Please try again.",
                }
            )
        except Exception:
            pass
