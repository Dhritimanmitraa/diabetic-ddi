"""
Vision OCR Service for Prescription Extraction.

Industry-standard approach:
1. EasyOCR - Extract text from prescription images (with preprocessing)
2. LLaMA - Structure extracted text into medicine objects

EasyOCR is used instead of PaddleOCR due to langchain dependency conflicts.
"""
import base64
import json
import logging
import re
import io
from typing import Optional, List, Tuple
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings
from app.prescription.schemas import MedicineCreate, ExtractionResult

logger = logging.getLogger(__name__)
settings = get_settings()


class ImagePreprocessor:
    """
    Light preprocessing to improve OCR accuracy for prescription images.
    
    Note: Minimal preprocessing for handwriting - too much can harm detection.
    """
    
    @staticmethod
    def preprocess(image_data: bytes) -> np.ndarray:
        """
        Apply light preprocessing for handwriting OCR.
        
        Steps:
        1. Decode image
        2. Convert to RGB (EasyOCR works better with color)
        3. Light resize if too small
        
        Note: Avoid aggressive preprocessing (contrast eq, blur) for handwriting
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Keep in RGB for better handwriting detection
        # EasyOCR handles color images well
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Light resize only if image is too small
        height, width = rgb.shape[:2]
        if max(height, width) < 1000:
            scale = 1.3
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return rgb


class EasyOCRExtractor:
    """
    EasyOCR-based text extraction with confidence filtering.
    
    Good OCR for prescriptions (printed + semi-handwritten).
    """
    
    def __init__(self):
        self._reader = None
        self._initialized = False
        self._preprocessor = ImagePreprocessor()
    
    def _init_ocr(self):
        """Lazy initialization of EasyOCR."""
        if self._initialized:
            return
        
        try:
            import easyocr
            
            # Initialize with English, use GPU if available
            self._reader = easyocr.Reader(['en'], gpu=False)
            self._initialized = True
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._initialized = False
    
    def extract_text(self, image_data: bytes, confidence_threshold: float = 0.15) -> Tuple[str, float]:
        """
        Extract text from image using EasyOCR.
        
        Args:
            image_data: Raw image bytes
            confidence_threshold: Minimum confidence to include text (0.3 for handwriting)
            
        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        self._init_ocr()
        
        if not self._reader:
            return "", 0.0
        
        try:
            # Preprocess image for better OCR
            processed_img = self._preprocessor.preprocess(image_data)
            
            # Run OCR
            results = self._reader.readtext(processed_img)
            
            if not results:
                logger.warning("EasyOCR returned empty result")
                return "", 0.0
            
            # Extract text with confidence filtering
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                # Only include text above confidence threshold
                if confidence >= confidence_threshold:
                    texts.append(text)
                    confidences.append(confidence)
                else:
                    logger.debug(f"Filtered low-confidence text: '{text}' ({confidence:.2f})")
            
            full_text = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(f"EasyOCR extracted {len(texts)} lines, avg confidence: {avg_confidence:.2f}")
            logger.info(f"OCR Text:\n{full_text}")
            
            return full_text, avg_confidence
            
        except Exception as e:
            logger.error(f"EasyOCR extraction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "", 0.0


class LLaMaMedicineParser:
    """
    Use LLaMA to structure OCR text into medicine objects.
    
    This is much more reliable than trying to use VLM directly on images.
    """
    
    PARSE_PROMPT = """You are a medical prescription parser. Extract medicines from this OCR text.

OCR Text from prescription:
{ocr_text}

Extract EVERY medicine mentioned. Common patterns:
- Tab/Tablet, Cap/Capsule, Syr/Syrup, Inj/Injection
- Dosages like 500mg, 250mg, 10ml
- Frequency like BD (twice daily), TDS (three times), OD (once daily)
- Timing like 1-0-1 (morning-afternoon-night)

Return ONLY a JSON array:
[
  {{"name": "medicine name", "generic_name": null, "quantity": null, "dosage": "dose if found", "frequency": "timing if found", "duration": null, "instructions": null, "morning": true, "afternoon": false, "evening": false, "night": true}}
]

If no medicines found, return: []
Extract all medicines now:"""

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
    
    def parse_to_medicines(self, ocr_text: str) -> List[MedicineCreate]:
        """Parse OCR text into structured medicine objects using LLaMA."""
        if not ocr_text or not ocr_text.strip():
            return []
        
        try:
            import ollama
            
            prompt = self.PARSE_PROMPT.format(ocr_text=ocr_text)
            
            logger.info(f"Sending to {self.model} for medicine extraction...")
            
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 2048}
            )
            
            response_text = response['message']['content']
            logger.info(f"LLaMA response:\n{response_text}")
            
            return self._parse_json_response(response_text)
            
        except Exception as e:
            logger.error(f"LLaMA parsing error: {e}")
            return []
    
    def _parse_json_response(self, text: str) -> List[MedicineCreate]:
        """Parse JSON from LLM response."""
        medicines = []
        
        try:
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'name' in item:
                            name = item.get('name', '')
                            # Skip placeholders
                            if not name or name.lower() in ['medicine name', '...', 'example']:
                                continue
                            
                            medicine = MedicineCreate(
                                name=name,
                                generic_name=item.get('generic_name'),
                                quantity=item.get('quantity'),
                                dosage=item.get('dosage'),
                                frequency=item.get('frequency'),
                                duration=item.get('duration'),
                                instructions=item.get('instructions'),
                                morning=item.get('morning', False),
                                afternoon=item.get('afternoon', False),
                                evening=item.get('evening', False),
                                night=item.get('night', False),
                            )
                            medicines.append(medicine)
                            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
        except Exception as e:
            logger.warning(f"Parse error: {e}")
        
        return medicines


class VisionOCRService:
    """
    Vision OCR Service using industry-standard approach:
    
    1. EasyOCR → Text extraction (with preprocessing + confidence filtering)
    2. LLaMA → Medicine structuring
    
    Much more reliable than pure VLM for handwritten prescriptions.
    """
    
    def __init__(self):
        self.ocr_extractor = EasyOCRExtractor()
        self.llm_parser = LLaMaMedicineParser(model="llama3.1:8b")
    
    async def extract_from_image(
        self, 
        image_data: bytes, 
        filename: str = "prescription.jpg"
    ) -> ExtractionResult:
        """
        Extract medicines from a prescription image.
        
        Uses EasyOCR + LLaMA hybrid approach for best results.
        """
        logger.info(f"Processing prescription: {filename}, size: {len(image_data)} bytes")
        
        # Step 1: Extract text using EasyOCR
        ocr_text, ocr_confidence = self.ocr_extractor.extract_text(image_data)
        
        if not ocr_text:
            logger.warning("EasyOCR failed to extract text")
            return ExtractionResult(
                raw_text="",
                medicines=[],
                confidence=0.0,
                model_used="easyocr-failed",
                error="OCR failed to extract text from image"
            )
        
        # Step 2: Parse text into medicines using LLaMA
        medicines = self.llm_parser.parse_to_medicines(ocr_text)
        
        if not medicines:
            logger.warning("LLaMA failed to extract medicines from OCR text")
            return ExtractionResult(
                raw_text=ocr_text,
                medicines=[],
                confidence=ocr_confidence * 0.5,
                model_used="easyocr+llama",
                error="No medicines extracted from OCR text"
            )
        
        logger.info(f"Successfully extracted {len(medicines)} medicines")
        
        return ExtractionResult(
            raw_text=ocr_text,
            medicines=medicines,
            confidence=ocr_confidence,
            model_used="easyocr+llama3.1",
            error=None
        )
    
    async def extract_from_pdf(self, pdf_data: bytes) -> ExtractionResult:
        """Extract medicines from a PDF prescription."""
        try:
            from pdf2image import convert_from_bytes
            
            images = convert_from_bytes(pdf_data, dpi=200)
            
            all_medicines = []
            all_text = []
            total_confidence = 0.0
            
            for i, image in enumerate(images):
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_data = img_buffer.getvalue()
                
                result = await self.extract_from_image(img_data, f"page_{i+1}.png")
                
                if result.medicines:
                    all_medicines.extend(result.medicines)
                if result.raw_text:
                    all_text.append(result.raw_text)
                total_confidence += result.confidence
            
            avg_confidence = total_confidence / len(images) if images else 0.0
            
            return ExtractionResult(
                raw_text="\n\n".join(all_text),
                medicines=all_medicines,
                confidence=avg_confidence,
                model_used="easyocr+llama3.1",
                error=None if all_medicines else "No medicines extracted from PDF"
            )
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ExtractionResult(
                raw_text="",
                medicines=[],
                confidence=0.0,
                model_used="error",
                error=str(e)
            )


# Singleton instance
_vision_service = None

def get_vision_service() -> VisionOCRService:
    """Get or create the vision OCR service singleton."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionOCRService()
    return _vision_service
