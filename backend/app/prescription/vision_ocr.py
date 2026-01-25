"""
Vision OCR Service for Prescription Extraction.

Updated Stack (Gemini Primary):
1. Gemini Vision (gemini-1.5-flash) - Primary OCR + Extraction
2. llama3.2-vision (Ollama) - Fallback OCR
3. llama3.1:8b (Ollama) - Fallback LLM
4. Regex patterns - Final fallback
"""
import base64
import json
import logging
import re
import io
from typing import Optional, List

from app.config import get_settings
from app.prescription.schemas import MedicineCreate, ExtractionResult

logger = logging.getLogger(__name__)
settings = get_settings()

# Primary: Gemini API
GEMINI_MODEL = "gemini-2.0-flash"  # Fast and capable vision model

# Fallback: Ollama models
FALLBACK_OCR_MODEL = "llama3.2-vision"  # Local VLM for OCR
FALLBACK_LLM = "llama3.1:8b"  # Local LLM

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

# Medicine extraction prompt (for Gemini - combined OCR + extraction)
GEMINI_EXTRACTION_PROMPT = """Analyze this prescription image and extract all medicines.

For each medicine found, provide:
- name: The medicine/drug name
- dosage: The dosage amount (e.g., 500mg, 10ml) or null
- frequency: How often to take it (e.g., BD, TDS, 1-0-1) or null

Return ONLY a valid JSON array like this:
[{"name": "Medicine Name", "dosage": "dose or null", "frequency": "timing or null"}]

If no medicines found, return: []"""

# Text extraction prompt
EXTRACTION_PROMPT = """You are a medical prescription parser. Extract all medicines from this text.

TEXT FROM PRESCRIPTION:
{ocr_text}

RULES:
1. Extract ONLY medicine/drug names
2. Include dosage if visible (500mg, 10ml, etc.)
3. Include frequency if visible (BD, TDS, OD, 1-0-1)
4. Ignore patient name, doctor name, dates, addresses

OUTPUT FORMAT - Return ONLY valid JSON array:
[{{"name": "Medicine Name", "dosage": "dose or null", "frequency": "timing or null"}}]

If no medicines found, return: []"""


class VisionOCRService:
    """Vision OCR using Gemini (primary) + Ollama (fallback)."""
    
    def __init__(self):
        self.gemini_model = None
        self._init_gemini()
        logger.info("VisionOCRService initialized (Gemini primary, Ollama fallback)")
    
    def _init_gemini(self):
        """Initialize Gemini Vision model."""
        try:
            import google.generativeai as genai
            
            api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
                logger.info(f"Gemini model initialized: {GEMINI_MODEL}")
            else:
                logger.warning("No Gemini API key found, will use Ollama fallback")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    
    async def extract_from_image(
        self, 
        image_data: bytes, 
        filename: str = "prescription.jpg"
    ) -> ExtractionResult:
        """Extract medicines from prescription image using Gemini (primary) or Ollama (fallback)."""
        logger.info(f"Processing: {filename}, size: {len(image_data)} bytes")
        
        medicines = []
        ocr_text = ""
        model_used = ""
        
        # Try Gemini first (combined OCR + extraction)
        if self.gemini_model:
            logger.info("Step 1: Trying Gemini Vision (primary)...")
            result = await self._gemini_extract(image_data)
            if result:
                medicines, ocr_text = result
                model_used = GEMINI_MODEL
                logger.info(f"Gemini extracted {len(medicines)} medicines")
        
        # Fallback to Ollama if Gemini failed
        if not medicines:
            logger.info("Step 2: Falling back to Ollama OCR...")
            ocr_text = await self._run_ocr(image_data)
            
            if ocr_text and len(ocr_text) >= 10:
                logger.info(f"OCR extracted {len(ocr_text)} chars")
                medicines = await self._extract_medicines(ocr_text)
                model_used = f"{FALLBACK_OCR_MODEL} + {FALLBACK_LLM}"
        
        # Final fallback: regex extraction
        if not medicines and ocr_text:
            logger.info("Step 3: Trying regex extraction...")
            medicines = self._regex_extract_medicines(ocr_text)
            model_used = model_used or "regex"
        
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
                confidence=0.90 if GEMINI_MODEL in model_used else 0.75,
                model_used=model_used,
                error=None
            )
        
        return ExtractionResult(
            raw_text=ocr_text or "[No text extracted]",
            medicines=[],
            confidence=0.0,
            model_used=model_used or "none",
            error="Could not extract medicines from image."
        )
    
    async def _gemini_extract(self, image_data: bytes) -> Optional[tuple]:
        """Use Gemini Vision for combined OCR + medicine extraction."""
        try:
            from PIL import Image
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Use Gemini to extract medicines directly
            response = self.gemini_model.generate_content([
                GEMINI_EXTRACTION_PROMPT,
                image
            ])
            
            if response and response.text:
                response_text = response.text.strip()
                logger.info(f"Gemini response: {response_text[:200]}...")
                
                # Parse JSON from response
                medicines = self._parse_medicines(response_text)
                return (medicines, response_text) if medicines else None
            
        except Exception as e:
            logger.error(f"Gemini extraction error: {e}")
        
        return None
    
    async def _run_ocr(self, image_data: bytes) -> str:
        """Run OCR with Ollama llama3.2-vision as fallback."""
        # Try local Ollama first
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
            logger.warning(f"Ollama OCR failed: {e}, trying Tesseract...")
        
        # Fallback: Tesseract OCR (fully local, no cloud)
        try:
            import pytesseract
            from PIL import Image
            import io
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Run Tesseract OCR
            text = pytesseract.image_to_string(image)
            
            if text and len(text.strip()) > 5:
                logger.info("Tesseract OCR successful")
                return text.strip()
            
        except ImportError:
            logger.error("Tesseract not installed. Install with: pip install pytesseract")
            logger.error("Also need Tesseract binary: https://github.com/tesseract-ocr/tesseract")
        except Exception as e:
            logger.error(f"Tesseract OCR also failed: {e}")
        
        return ""
    
    async def _extract_medicines(self, ocr_text: str) -> List[MedicineCreate]:
        """Extract medicines from OCR text using LLM with regex fallback."""
        medicines = []
        
        # Try LLM first
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
            
            # Check if LLM refused
            refusal_phrases = ["cannot provide", "cannot help", "not able to", "unable to", "i can't"]
            if any(phrase in response_text.lower() for phrase in refusal_phrases):
                logger.warning("LLM refused to parse, using regex fallback...")
                medicines = self._regex_extract_medicines(ocr_text)
            else:
                # Parse JSON response
                medicines = self._parse_medicines(response_text)
            
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
        
        # If LLM extraction failed, try regex
        if not medicines:
            logger.info("Trying regex-based medicine extraction...")
            medicines = self._regex_extract_medicines(ocr_text)
        
        return medicines
    
    def _regex_extract_medicines(self, text: str) -> List[MedicineCreate]:
        """Extract medicines using regex patterns (fallback method)."""
        medicines = []
        
        # Common medicine patterns
        patterns = [
            # "Medicine Name: dosage" or "Medicine Name - dosage"
            r'[•\*\-]\s*([A-Z][a-zA-Z\s]+?)(?:[:,\-]\s*)(\d+(?:\.\d+)?(?:\s*(?:mg|ml|gm|gram|capsule|tablet|cap|tab|spanule)s?)?)',
            # Lines starting with medicine names followed by quantity
            r'(?:^|\n)\s*([A-Z][a-zA-Z]+(?:\s[A-Z]?[a-zA-Z]+)?)\s*[:,-]?\s*(\d+(?:\s*[-x]\s*\d+)?(?:\s*(?:mg|ml|gm|gram|capsule|tablet|cap|tab)s?)?)',
            # "Rx:" followed by medicine
            r'(?:Rx|Tab|Cap|Syp)[:\s]+([A-Z][a-zA-Z\s]+?)(?:\s+)(\d+[^\n]*)',
        ]
        
        found_names = set()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                dosage = match.group(2).strip() if len(match.groups()) > 1 else None
                
                # Filter out common non-medicine words
                skip_words = ['name', 'patient', 'address', 'date', 'doctor', 'dr', 'prescription', 
                             'information', 'instructions', 'the', 'for', 'take', 'with']
                if name.lower() in skip_words or len(name) < 3:
                    continue
                
                # Avoid duplicates
                if name.lower() not in found_names:
                    found_names.add(name.lower())
                    medicines.append(MedicineCreate(
                        name=name.title(),
                        dosage=dosage,
                        frequency=None
                    ))
        
        logger.info(f"Regex extraction found {len(medicines)} medicines")
        return medicines
    
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
