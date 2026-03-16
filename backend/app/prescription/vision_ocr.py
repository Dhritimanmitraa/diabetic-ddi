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
from app.services.gemini_client import get_gemini_client
from app.services.nvidia_vision_client import get_nvidia_vision_client

logger = logging.getLogger(__name__)
settings = get_settings()

# Primary: Gemini API
GEMINI_MODEL = "gemini-2.0-flash"  # Fast and capable vision model

# Fallback: Ollama models (from settings)
FALLBACK_OCR_MODEL = settings.OLLAMA_VISION_MODEL
FALLBACK_LLM = settings.OLLAMA_MODEL
OCR_MODEL = FALLBACK_OCR_MODEL
LLM_MODEL = FALLBACK_LLM

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
    """Vision OCR using NVIDIA Cosmos (primary) + Gemini + Ollama (fallback)."""
    
    def __init__(self):
        self.gemini_model = None
        self.nvidia_client = None
        self._init_nvidia()
        self._init_gemini()
        logger.info("VisionOCRService initialized (NVIDIA primary, Gemini secondary, Ollama fallback)")
    
    def _init_nvidia(self):
        """Initialize NVIDIA Cosmos Vision model."""
        self.nvidia_client = get_nvidia_vision_client()
        if self.nvidia_client.is_available:
            logger.info(f"NVIDIA Cosmos model initialized: {settings.NVIDIA_COSMOS_MODEL}")
        else:
            logger.warning("No NVIDIA API key found, will skip NVIDIA Cosmos")
    
    @property
    def nvidia_available(self) -> bool:
        """Check if NVIDIA Cosmos is available."""
        return self.nvidia_client is not None and self.nvidia_client.is_available
    
    @property
    def gemini_available(self) -> bool:
        """Check if Gemini is available."""
        return self.gemini_model is not None and self.gemini_model.is_available
    
    def _init_gemini(self):
        """Initialize Gemini Vision model."""
        self.gemini_model = get_gemini_client(GEMINI_MODEL)
        if self.gemini_model.is_available:
            logger.info(f"Gemini model initialized: {GEMINI_MODEL} via {self.gemini_model.sdk}")
        else:
            logger.warning("No Gemini API key found, will use Ollama fallback")
    
    async def extract_from_image(
        self, 
        image_data: bytes, 
        filename: str = "prescription.jpg"
    ) -> ExtractionResult:
        """Extract medicines from prescription image using NVIDIA (primary), Gemini, or Ollama (fallback)."""
        logger.info(f"Processing: {filename}, size: {len(image_data)} bytes")
        
        medicines = []
        ocr_text = ""
        model_used = ""
        
        # Try NVIDIA Cosmos first (combined OCR + extraction)
        if self.nvidia_available:
            logger.info("Step 1: Trying NVIDIA Cosmos Vision (primary)...")
            result = await self._nvidia_extract(image_data)
            if result:
                medicines, ocr_text = result
                model_used = settings.NVIDIA_COSMOS_MODEL
                logger.info(f"NVIDIA Cosmos extracted {len(medicines)} medicines")
        
        # Try Gemini second (combined OCR + extraction)
        if not medicines and self.gemini_model and self.gemini_model.is_available:
            logger.info("Step 2: Trying Gemini Vision (secondary)...")
            result = await self._gemini_extract(image_data)
            if result:
                medicines, ocr_text = result
                model_used = GEMINI_MODEL
                logger.info(f"Gemini extracted {len(medicines)} medicines")
        
        # Fallback to Ollama if NVIDIA and Gemini both failed
        if not medicines:
            logger.info("Step 3: Falling back to Ollama OCR...")
            ocr_text = await self._run_ocr(image_data)
            
            if ocr_text and len(ocr_text) >= 10:
                logger.info(f"OCR extracted {len(ocr_text)} chars")
                medicines = await self._extract_medicines(ocr_text)
                model_used = f"{FALLBACK_OCR_MODEL} + {FALLBACK_LLM}"
        
        # Final fallback: regex extraction
        if not medicines and ocr_text:
            logger.info("Step 4: Trying regex extraction...")
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

    async def extract_from_pdf(self, pdf_data: bytes) -> ExtractionResult:
        """Extract medicines from a PDF via embedded text first, then page images."""
        logger.info(f"Processing PDF, size: {len(pdf_data)} bytes")

        pdf_text = self._extract_text_from_pdf(pdf_data)
        if pdf_text:
            medicines = await self._extract_medicines(pdf_text)
            if not medicines:
                medicines = self._regex_extract_medicines(pdf_text)

            if medicines:
                for med in medicines:
                    med.name = self._normalize_drug(med.name)

                raw_text = "\n".join(
                    f"- {m.name} {m.dosage or ''} {m.frequency or ''}".strip()
                    for m in medicines
                )
                return ExtractionResult(
                    raw_text=raw_text,
                    medicines=medicines,
                    confidence=0.85,
                    model_used="pypdf2 + llm",
                    error=None,
                )

        images = self._convert_pdf_to_images(pdf_data)
        if images:
            page_results = []
            collected_text = []
            for index, image_bytes in enumerate(images[:3], start=1):
                result = await self.extract_from_image(image_bytes, f"prescription_page_{index}.png")
                if result.medicines:
                    page_results.extend(result.medicines)
                if result.raw_text:
                    collected_text.append(result.raw_text)

            deduped = self._dedupe_medicines(page_results)
            if deduped:
                return ExtractionResult(
                    raw_text="\n\n".join(collected_text) or "[PDF converted to images]",
                    medicines=deduped,
                    confidence=0.80,
                    model_used="pdf2image + vision",
                    error=None,
                )

        return ExtractionResult(
            raw_text=pdf_text or "[No text extracted from PDF]",
            medicines=[],
            confidence=0.0,
            model_used="pdf",
            error="Could not extract medicines from PDF.",
        )

    def _extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Extract text from PDF using a text parser when available."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(pdf_data))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text.strip())
            return "\n\n".join(pages).strip()
        except Exception as exc:
            logger.warning(f"Direct PDF text extraction failed: {exc}")
            return ""

    def _convert_pdf_to_images(self, pdf_data: bytes) -> List[bytes]:
        """Convert PDF pages to PNG images for OCR fallback."""
        try:
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(pdf_data, fmt="png", dpi=200, first_page=1, last_page=3)
            output = []
            for image in images:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                output.append(buffer.getvalue())
            return output
        except Exception as exc:
            logger.warning(f"PDF to image conversion failed: {exc}")
            return []

    def _dedupe_medicines(self, medicines: List[MedicineCreate]) -> List[MedicineCreate]:
        """Remove duplicate medicines after multi-page PDF extraction."""
        unique: dict[tuple[str, Optional[str], Optional[str]], MedicineCreate] = {}
        for medicine in medicines:
            key = (
                medicine.name.strip().lower(),
                (medicine.dosage or "").strip().lower() or None,
                (medicine.frequency or "").strip().lower() or None,
            )
            unique.setdefault(key, medicine)
        return list(unique.values())
    
    async def _nvidia_extract(self, image_data: bytes) -> Optional[tuple]:
        """Use NVIDIA Cosmos Reason2-8B for combined OCR + medicine extraction."""
        try:
            from PIL import Image
            
            # Convert bytes to PIL Image and normalize to PNG
            image = Image.open(io.BytesIO(image_data))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()
            
            response = self.nvidia_client.generate_from_image(
                GEMINI_EXTRACTION_PROMPT,  # Reuse the same extraction prompt
                image_bytes=png_bytes,
                mime_type="image/png",
                temperature=0.1,
                max_tokens=1200,
            )
            
            if response and response.text:
                response_text = response.text.strip()
                logger.info(f"NVIDIA Cosmos response: {response_text[:200]}...")
                
                # Parse JSON from response
                medicines = self._parse_medicines(response_text)
                return (medicines, response_text) if medicines else None
            
        except Exception as e:
            logger.error(f"NVIDIA Cosmos extraction error: {e}")
        
        return None
    
    async def _gemini_extract(self, image_data: bytes) -> Optional[tuple]:
        """Use Gemini Vision for combined OCR + medicine extraction."""
        try:
            from PIL import Image
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Use Gemini to extract medicines directly
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            response = self.gemini_model.generate_with_media(
                GEMINI_EXTRACTION_PROMPT,
                media_bytes=buffer.getvalue(),
                mime_type="image/png",
                temperature=0.1,
                max_output_tokens=1200,
            )
            
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
        prompt = EXTRACTION_PROMPT.format(ocr_text=ocr_text)
        medicines = []
        refusal_phrases = ["cannot provide", "cannot help", "not able to", "unable to", "i can't"]
        
        # 1. Try NVIDIA Cosmos text API first
        if self.nvidia_available:
            try:
                logger.info("Trying NVIDIA Cosmos for text extraction...")
                response = self.nvidia_client.generate_text(
                    prompt, 
                    temperature=0.1, 
                    max_tokens=2048
                )
                if response and response.text:
                    if not any(phrase in response.text.lower() for phrase in refusal_phrases):
                        medicines = self._parse_medicines(response.text)
                        if medicines:
                            return medicines
            except Exception as e:
                logger.warning(f"NVIDIA text extraction failed: {e}")

        # 2. Try Gemini Text API second
        if self.gemini_model and self.gemini_model.is_available:
            try:
                logger.info("Trying Gemini for text extraction...")
                response = self.gemini_model.generate_text(
                    prompt,
                    temperature=0.1,
                    max_output_tokens=2048
                )
                if response and response.text:
                    if not any(phrase in response.text.lower() for phrase in refusal_phrases):
                        medicines = self._parse_medicines(response.text)
                        if medicines:
                            return medicines
            except Exception as e:
                logger.warning(f"Gemini text extraction failed: {e}")

        # 3. Try Ollama (Local LLM)
        try:
            import ollama
            
            ollama_model = settings.OLLAMA_MODEL
            logger.info(f"Trying local Ollama model: {ollama_model}...")
            
            response = ollama.chat(
                model=ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 2048}
            )
            response_text = response['message']['content']
            
            if not any(phrase in response_text.lower() for phrase in refusal_phrases):
                medicines = self._parse_medicines(response_text)
                if medicines:
                    return medicines
        except Exception as e:
            logger.warning(f"Ollama extraction error: {e}")
        
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
