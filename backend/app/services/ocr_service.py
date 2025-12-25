"""
OCR Service for extracting drug names from images.

Uses OpenCV for image preprocessing and Tesseract for OCR.
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
import base64
import io
import re
from typing import List, Tuple, Optional
import logging
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugOCRService:
    """
    Service for extracting drug names from images using OCR.
    
    Processes medication labels, prescription bottles, and drug packaging.
    """
    
    # Common drug name patterns and keywords
    DRUG_KEYWORDS = [
        "tablet", "tablets", "capsule", "capsules", "mg", "ml", "mcg", "injection", "syrup",
        "suspension", "cream", "ointment", "gel", "drops", "solution",
        "extended release", "er", "sr", "xr", "cr", "dr", "succinate", "tartrate",
        "hydrochloride", "hcl", "acetate", "sulfate", "phosphate"
    ]
    
    # Regex patterns for drug names - EXPANDED
    DRUG_PATTERNS = [
        # Common drug suffixes (generic names)
        r'\b[A-Z][a-z]+(?:in|ol|am|an|ide|ate|ine|one|ril|pril|sartan|statin|azole|mycin|floxacin|cillin|gliflozin|gliptin|glutide|xaban|gatran|codone|dronate|prazole|tidine|dipine|olol|zepam|zolam)\b',
        # Brand names (ALL CAPS, often 4-10 chars)
        r'\b[A-Z]{3,10}(?:[-]?[A-Z0-9]+)?\b',  # TEKSAN, TELNOX-M25, etc.
        # Generic drug names (common ones)
        r'\b(?:acetaminophen|ibuprofen|aspirin|metformin|lisinopril|omeprazole|metoprolol|amlodipine|atorvastatin|simvastatin|levothyroxine|albuterol|prednisone|gabapentin|tramadol|sertraline|fluoxetine|amoxicillin|azithromycin|doxycycline|ciprofloxacin|warfarin|clopidogrel|furosemide|hydrochlorothiazide|losartan|valsartan|carvedilol|propranolol|atenolol|diltiazem|verapamil|digoxin|spironolactone|metronidazole|cephalexin|clindamycin|nitrofurantoin|trimethoprim|sulfamethoxazole)\b',
        # Drug names with dosage (e.g., "Metoprolol 25mg")
        r'\b[A-Z][a-z]+\s+(?:Succinate|Tartrate|Hydrochloride|Acetate|Sulfate|Phosphate)?\s*(?:ER|SR|XR|CR|DR|Extended Release)?\s*(?:\d+\s*(?:mg|mcg|ml))?\b',
        # ALL CAPS drug names with dosage
        r'\b[A-Z]{2,}(?:[-]?[A-Z0-9]+)?(?:\s+\d+\s*(?:mg|mcg|ml))?\b',
        # Words before "Tablet", "Capsule", etc.
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Tablet|Capsule|Injection|Syrup|Suspension|Cream|Ointment|Gel|Drops|Solution)\b',
    ]
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the OCR service.
        
        Args:
            tesseract_cmd: Path to tesseract executable (Windows)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Applies various image processing techniques to enhance text visibility.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def preprocess_for_medication_label(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Special preprocessing for medication labels.
        
        Returns multiple processed versions for better OCR coverage.
        """
        processed_images = []
        
        # Original grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        processed_images.append(gray)
        
        # Standard preprocessing
        processed_images.append(self.preprocess_image(image))
        
        # High contrast version (OTSU thresholding)
        _, high_contrast = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(high_contrast)
        
        # Inverted version (for dark backgrounds)
        inverted = cv2.bitwise_not(gray)
        _, inverted_thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(inverted_thresh)
        
        # Rescaled version (for small text) - 2x scale
        scale_percent = 200
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        processed_images.append(resized)
        
        # Sharpen the image for better text recognition
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        processed_images.append(sharpened)
        
        # Bilateral filter for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        processed_images.append(bilateral)
        
        # Morphological operations to enhance text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morph)
        
        return processed_images
    
    def extract_text(self, image: np.ndarray, config: str = '--oem 3 --psm 6') -> str:
        """
        Extract text from preprocessed image using Tesseract.
        
        Args:
            image: Preprocessed image array
            config: Tesseract configuration string
            
        Returns:
            Extracted text
        """
        try:
            # Use both image_to_string and image_to_data for better results
            text = pytesseract.image_to_string(image, config=config)
            
            # Also get detailed data for confidence scoring
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Filter out low-confidence words (optional - can be used for better filtering)
            # For now, just return the text
            
            return text.strip()
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def extract_from_base64(self, base64_string: str) -> Tuple[str, List[str], float]:
        """
        Extract drug names from a base64 encoded image.
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            Tuple of (raw_text, detected_drugs, confidence_score)
        """
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return "", [], 0.0
        
        return self.extract_drugs_from_image(image_array)
    
    def extract_drugs_from_image(self, image: np.ndarray) -> Tuple[str, List[str], float]:
        """
        Extract drug names from an image.
        
        Args:
            image: Image array (BGR or grayscale)
            
        Returns:
            Tuple of (raw_text, detected_drugs, confidence_score)
        """
        all_texts = []
        
        # Get multiple preprocessed versions
        preprocessed_images = self.preprocess_for_medication_label(image)
        
        # Try different Tesseract configurations - optimized for medication labels
        configs = [
            '--oem 3 --psm 6',   # Assume uniform block of text
            '--oem 3 --psm 3',   # Fully automatic page segmentation
            '--oem 3 --psm 11',  # Sparse text (good for labels)
            '--oem 3 --psm 4',   # Single column of text
            '--oem 3 --psm 7',   # Single text line
            '--oem 3 --psm 8',   # Single word
            '--oem 3 --psm 12',  # Sparse text with OSD
        ]
        
        # Extract text from all versions
        for img in preprocessed_images:
            for config in configs:
                text = self.extract_text(img, config)
                if text:
                    all_texts.append(text)
        
        # Combine all extracted texts
        combined_text = '\n'.join(all_texts)
        
        # Extract drug names
        detected_drugs = self._extract_drug_names(combined_text)
        
        # Calculate confidence based on consistency across extractions
        confidence = self._calculate_confidence(all_texts, detected_drugs)
        
        return combined_text, detected_drugs, confidence
    
    def _extract_drug_names(self, text: str) -> List[str]:
        """
        Extract potential drug names from OCR text.
        
        Uses pattern matching and keyword detection.
        """
        drugs = set()
        
        # Clean text but preserve structure
        original_text = text
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Log extracted text for debugging
        logger.info(f"OCR extracted text (first 500 chars): {text[:500]}")
        
        # Apply regex patterns
        for pattern_idx, pattern in enumerate(self.DRUG_PATTERNS):
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Handle tuple matches (from groups)
                if isinstance(match, tuple):
                    match = ' '.join([m for m in match if m])
                
                cleaned = match.strip().upper()
                # Remove common suffixes/prefixes
                cleaned = cleaned.replace(' TABLET', '').replace(' TABLETS', '')
                cleaned = cleaned.replace(' CAPSULE', '').replace(' CAPSULES', '')
                cleaned = cleaned.replace(' ER', '').replace(' SR', '').replace(' XR', '')
                
                if len(cleaned) > 2:
                    drugs.add(cleaned)
        
        # Look for words near drug-related keywords
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,;:()[]')
            
            # Check if word is near a drug keyword
            for keyword in self.DRUG_KEYWORDS:
                if keyword in word_lower:
                    # Get surrounding words as potential drug names
                    start = max(0, i - 3)
                    end = min(len(words), i + 2)
                    
                    for j in range(start, end):
                        candidate = words[j].strip('.,;:()[]&')
                        # More lenient: allow numbers and hyphens (e.g., "TELNOX-M25")
                        if len(candidate) > 2:
                            # Allow alphanumeric and hyphens
                            if candidate.replace('-', '').replace('&', '').isalnum():
                                drugs.add(candidate.upper())
        
        # Extract words that look like drug names (capitalized, 4+ chars)
        # This catches brand names like "TEKSAN", "TELNOX"
        words = original_text.split()
        for word in words:
            cleaned = word.strip('.,;:()[]&').upper()
            # Brand names: all caps, 3-12 chars, may have numbers/hyphens
            if (cleaned.isupper() and 3 <= len(cleaned) <= 12 and 
                cleaned.replace('-', '').replace('&', '').isalnum() and
                not cleaned.isdigit()):
                drugs.add(cleaned)
        
        # Filter out common non-drug words
        non_drug_words = {
            'THE', 'AND', 'FOR', 'WITH', 'TAKE', 'USE', 'EACH', 'ONE',
            'TWO', 'THREE', 'DAILY', 'ONCE', 'TWICE', 'BEFORE', 'AFTER',
            'FOOD', 'WATER', 'DOSE', 'DOSAGE', 'WARNING', 'CAUTION',
            'STORE', 'KEEP', 'CHILDREN', 'REACH', 'PHARMACY', 'REFILL',
            'MADE', 'INDIA', 'MFG', 'EXP', 'INCL', 'TAXES', 'RS', 'PER',
            'TABS', 'TAB', 'CAP', 'CAPS', 'MRP', 'M.R.P.'
        }
        
        filtered_drugs = []
        for d in drugs:
            # Remove if it's a non-drug word
            if d in non_drug_words:
                continue
            # Remove if it's just numbers
            if d.isdigit():
                continue
            # Keep if it's 3+ chars
            if len(d) >= 3:
                filtered_drugs.append(d)
        
        # Remove duplicates and sort by length (longer = more likely to be drug name)
        filtered_drugs = sorted(list(set(filtered_drugs)), key=len, reverse=True)
        
        logger.info(f"Extracted potential drug names: {filtered_drugs[:10]}")
        
        return filtered_drugs[:20]  # Return top 20 candidates
    
    def _calculate_confidence(self, texts: List[str], drugs: List[str]) -> float:
        """
        Calculate confidence score based on OCR consistency.
        """
        if not drugs:
            return 0.0
        
        if not texts:
            return 0.0
        
        # Count how many texts contain each drug
        drug_counts = {}
        for drug in drugs:
            count = sum(1 for text in texts if drug.lower() in text.lower())
            drug_counts[drug] = count
        
        # Calculate average detection rate
        avg_detection = sum(drug_counts.values()) / (len(drugs) * len(texts))
        
        # Base confidence
        confidence = min(avg_detection * 1.5, 1.0)
        
        # Boost confidence for longer drug names (less likely to be false positives)
        avg_length = sum(len(d) for d in drugs) / len(drugs)
        if avg_length > 8:
            confidence = min(confidence + 0.1, 1.0)
        
        return round(confidence, 2)
    
    def find_similar_drug_names(self, detected_name: str, known_drugs: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find similar drug names in the known drug list.
        
        Handles OCR errors and partial matches.
        
        Args:
            detected_name: Name detected from OCR
            known_drugs: List of known drug names
            threshold: Minimum similarity score
            
        Returns:
            List of (drug_name, similarity_score) tuples
        """
        matches = []
        detected_upper = detected_name.upper()
        
        for drug in known_drugs:
            drug_upper = drug.upper()
            
            # Exact match
            if detected_upper == drug_upper:
                matches.append((drug, 1.0))
                continue
            
            # Check if one contains the other
            if detected_upper in drug_upper or drug_upper in detected_upper:
                score = len(min(detected_upper, drug_upper)) / len(max(detected_upper, drug_upper))
                if score >= threshold:
                    matches.append((drug, score))
                continue
            
            # Use sequence matcher for fuzzy matching
            ratio = SequenceMatcher(None, detected_upper, drug_upper).ratio()
            if ratio >= threshold:
                matches.append((drug, ratio))
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:5]  # Return top 5 matches


def create_ocr_service(tesseract_cmd: Optional[str] = None) -> DrugOCRService:
    """Factory function to create OCR service."""
    return DrugOCRService(tesseract_cmd)

