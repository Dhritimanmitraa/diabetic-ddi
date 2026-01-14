"""
Vision OCR Service for Prescription Extraction.

Simplified Stack:
1. llama3.2-vision (Ollama) - Primary OCR  
2. gemini-3-flash-preview:cloud (Ollama) - Extract medicines from OCR text
3. RapidFuzz - Drug name normalization
4. llama3.1:8b - Local LLM fallback
"""
import base64
import json
import logging
import re
from typing import Optional, List

from app.config import get_settings
from app.prescription.schemas import MedicineCreate, ExtractionResult

logger = logging.getLogger(__name__)
settings = get_settings()

# Models
OCR_MODEL = "llama3.2-vision"  # Working VLM for OCR
LLM_MODEL = "gemini-3-flash-preview:cloud"  # Cloud LLM for extraction
FALLBACK_LLM = "llama3.1:8b"  # Local fallback

# Common drug abbreviations for normalization
DRUG_ABBREVIATIONS = {
    "pcm": "Paracetamol",
    "para": "Paracetamol",
    "azee": "Azithromycin",
    "pan d": "Pantoprazole + Domperidone",
    "pan-d": "Pantoprazole + Domperidone",
    "met xl": "Metoprolol XL",
    "crocin": "Paracetamol",
    "dolo": "Paracetamol 650mg",
    "combiflam": "Ibuprofen + Paracetamol",
    "allegra": "Fexofenadine",
    "montair": "Montelukast",
    "montek": "Montelukast",
    "pantocid": "Pantoprazole",
    "omez": "Omeprazole",
    "rantac": "Ranitidine",
    "taxim": "Cefixime",
    "oflox": "Ofloxacin",
    "metrogyl": "Metronidazole",
    "telma": "Telmisartan",
    "atorva": "Atorvastatin",
    "ecosprin": "Aspirin",
}

# OCR Prompt
OCR_PROMPT = """Look at this prescription image and read ALL the text you can see.
Focus on medicine names, dosages, and instructions.
Output the raw text exactly as written."""

# Medicine extraction prompt
EXTRACTION_PROMPT = """You are a medical prescription parser. Extract all medicines from this text.

TEXT FROM PRESCRIPTION:
{ocr_text}

RULES:
1. Extract ONLY medicine/drug names
2. Include dosage if visible (500mg, 10ml, etc.)
3. Include frequency if visible (BD, TDS, OD, 1-0-1)
4. Ignore patient name, doctor name, dates, addresses

OUTPUT FORMAT - Return ONLY valid JSON array:
[
  {{"name": "Medicine Name", "dosage": "dose or null", "frequency": "timing or null"}}
]

If no medicines found, return: []"""


class VisionOCRService:
    """Vision OCR using llama3.2-vision + gemini-3-flash for extraction."""
    
    def __init__(self):
        logger.info("VisionOCRService initialized (llama3.2-vision + gemini)")
    
    async def extract_from_image(
        self, 
        image_data: bytes, 
        filename: str = "prescription.jpg"
    ) -> ExtractionResult:
        """Extract medicines from prescription image."""
        logger.info(f"Processing: {filename}, size: {len(image_data)} bytes")
        
        # Step 1: OCR with llama3.2-vision
        logger.info("Step 1: Running llama3.2-vision OCR...")
        ocr_text = await self._run_ocr(image_data)
        
        if not ocr_text or len(ocr_text) < 10:
            return ExtractionResult(
                raw_text="[No text extracted]",
                medicines=[],
                confidence=0.0,
                model_used=OCR_MODEL,
                error="Could not read text from image."
            )
        
        logger.info(f"OCR extracted {len(ocr_text)} chars: {ocr_text[:150]}...")
        
        # Step 2: Extract medicines with LLM
        logger.info("Step 2: Extracting medicines with LLM...")
        medicines = await self._extract_medicines(ocr_text)
        
        if medicines:
            # Normalize drug names
            for med in medicines:
                med.name = self._normalize_drug(med.name)
            
            raw_text = "\n".join([
                f"- {m.name} {m.dosage or ''} {m.frequency or ''}".strip() 
                for m in medicines
            ])
            
            return ExtractionResult(
                raw_text=raw_text,
                medicines=medicines,
                confidence=0.85,
                model_used=f"{OCR_MODEL} + {LLM_MODEL}",
                error=None
            )
        
        return ExtractionResult(
            raw_text=ocr_text,
            medicines=[],
            confidence=0.5,
            model_used=f"{OCR_MODEL} + {LLM_MODEL}",
            error="No medicines found in extracted text."
        )
    
    async def _run_ocr(self, image_data: bytes) -> str:
        """Run OCR with llama3.2-vision."""
        try:
            import ollama
            
            img_base64 = base64.b64encode(image_data).decode('utf-8')
            
            response = ollama.chat(
                model=OCR_MODEL,
                messages=[{
                    'role': 'user',
                    'content': OCR_PROMPT,
                    'images': [img_base64]
                }],
                options={'temperature': 0.1, 'num_predict': 2048},
                keep_alive='5m'
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    async def _extract_medicines(self, ocr_text: str) -> List[MedicineCreate]:
        """Extract medicines from OCR text using LLM."""
        try:
            import ollama
            
            prompt = EXTRACTION_PROMPT.format(ocr_text=ocr_text)
            
            # Try cloud model first
            try:
                logger.info(f"Trying {LLM_MODEL}...")
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.1, 'num_predict': 2048}
                )
                response_text = response['message']['content']
                
            except Exception as e:
                logger.warning(f"Cloud model failed: {e}, trying {FALLBACK_LLM}...")
                response = ollama.chat(
                    model=FALLBACK_LLM,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.1, 'num_predict': 2048}
                )
                response_text = response['message']['content']
            
            logger.info(f"LLM response: {response_text[:200]}...")
            
            # Parse JSON response
            return self._parse_medicines(response_text)
            
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return []
    
    def _parse_medicines(self, response_text: str) -> List[MedicineCreate]:
        """Parse LLM response into medicine objects."""
        try:
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                return []
            
            json_str = json_match.group()
            data = json.loads(json_str)
            
            medicines = []
            for item in data:
                if isinstance(item, dict) and 'name' in item:
                    name = item.get('name', '').strip()
                    if name and len(name) > 1:
                        medicines.append(MedicineCreate(
                            name=name,
                            dosage=item.get('dosage'),
                            frequency=item.get('frequency')
                        ))
            
            return medicines
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []
    
    def _normalize_drug(self, name: str) -> str:
        """Normalize drug name using abbreviations."""
        if not name:
            return name
        
        name_lower = name.lower().strip()
        
        # Check abbreviations
        if name_lower in DRUG_ABBREVIATIONS:
            return DRUG_ABBREVIATIONS[name_lower]
        
        # Try fuzzy match
        try:
            from rapidfuzz import process, fuzz
            
            match = process.extractOne(
                name_lower,
                list(DRUG_ABBREVIATIONS.keys()),
                scorer=fuzz.ratio
            )
            
            if match and match[1] >= 85:
                return DRUG_ABBREVIATIONS[match[0]]
                
        except ImportError:
            pass
        
        return name.title()


# Singleton
_vision_service = None

def get_vision_service() -> VisionOCRService:
    """Get vision service singleton."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionOCRService()
    return _vision_service
