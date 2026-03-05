from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import logging
import base64

from app.database import get_db
from app.models import Drug
from app.schemas import OCRRequest, OCRResponse
from app.services import create_ocr_service, create_interaction_service
from app.config import get_settings

router = APIRouter(
    tags=["OCR"]
)

logger = logging.getLogger(__name__)
settings = get_settings()

@router.post("/ocr/extract", response_model=OCRResponse)
async def extract_from_image(
    request: OCRRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract drug names from an image using OCR.
    
    Accepts base64 encoded image of medication labels, bottles, or prescriptions.
    """
    ocr_service = create_ocr_service(settings.TESSERACT_CMD)
    
    try:
        raw_text, detected_drugs, confidence = ocr_service.extract_from_base64(
            request.image_base64
        )
        
        # Try to match detected drugs with database with fuzzy matching
        service = create_interaction_service(db)
        matched_drugs = []
        
        logger.info(f"Attempting to match {len(detected_drugs)} detected drug names with database")
        
        for drug_name in detected_drugs:
            drugs = await service.search_drugs(drug_name, limit=5)
            
            if drugs:
                matched_drugs.append(drugs[0].name)
                logger.info(f"Matched '{drug_name}' -> '{drugs[0].name}'")
            else:
                # Narrow fuzzy matching to drugs containing any token from the OCR text
                tokens = [t for t in drug_name.upper().split() if len(t) >= 3]
                candidates = []
                if tokens:
                    token_conditions = [func.upper(Drug.name).contains(t) for t in tokens[:3]]
                    from sqlalchemy import or_
                    stmt = select(Drug.name).where(or_(*token_conditions)).limit(200)
                    result = await db.execute(stmt)
                    candidates = [d[0] for d in result.fetchall()]
                
                if candidates:
                    fuzzy_matches = ocr_service.find_similar_drug_names(
                        drug_name, candidates, threshold=0.5
                    )
                    if fuzzy_matches:
                        matched_drugs.append(fuzzy_matches[0][0])
                        logger.info(f"Fuzzy matched '{drug_name}' -> '{fuzzy_matches[0][0]}' (score: {fuzzy_matches[0][1]:.2f})")
                    else:
                        matched_drugs.append(drug_name)
                        logger.warning(f"No match found for '{drug_name}'")
                else:
                    matched_drugs.append(drug_name)
        
        return OCRResponse(
            extracted_text=raw_text[:1000],  # Limit text length
            detected_drugs=matched_drugs,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/ocr/upload", response_model=OCRResponse)
async def extract_from_upload(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Extract drug names from uploaded image file.
    
    Accepts image files (JPEG, PNG).
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Read and encode image
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    # Process using the base64 endpoint
    request = OCRRequest(image_base64=base64_image)
    return await extract_from_image(request, db)
