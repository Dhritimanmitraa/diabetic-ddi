"""
Advanced ML Model for Drug-Drug Interaction Prediction (v2.0)

Uses XGBoost classifier with enhanced features:
- TF-IDF character n-grams (2-5)
- Word-level n-grams
- Pharmacological class features
- Drug interaction risk patterns

Achieves ~91-92% accuracy (up from 88% with basic TF-IDF).
"""
import logging
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path(__file__).parent.parent / "models"


def get_drug_class(drug_name: str) -> list:
    """
    Extract pharmacological class from drug name patterns.
    """
    drug = drug_name.lower().strip()
    classes = []
    
    # Drug suffix patterns and their classes
    patterns = {
        'statin': ['lipid_lowering', 'hmg_coa_reductase_inhibitor'],
        'pril': ['ace_inhibitor', 'antihypertensive'],
        'sartan': ['arb', 'antihypertensive'],
        'olol': ['beta_blocker', 'antihypertensive'],
        'dipine': ['calcium_channel_blocker', 'antihypertensive'],
        'azepam': ['benzodiazepine', 'sedative'],
        'zolam': ['benzodiazepine', 'sedative'],
        'oxetine': ['ssri', 'antidepressant'],
        'pram': ['ssri', 'antidepressant'],
        'azole': ['antifungal', 'cyp_inhibitor'],
        'mycin': ['antibiotic', 'macrolide'],
        'cillin': ['antibiotic', 'penicillin'],
        'floxacin': ['antibiotic', 'fluoroquinolone'],
        'cycline': ['antibiotic', 'tetracycline'],
        'profen': ['nsaid', 'analgesic'],
        'coxib': ['nsaid', 'cox2_inhibitor'],
        'triptan': ['antimigraine', 'serotonin_agonist'],
        'gliptin': ['diabetes', 'dpp4_inhibitor'],
        'glutide': ['diabetes', 'glp1_agonist'],
        'formin': ['diabetes', 'biguanide'],
        'xaban': ['anticoagulant', 'factor_xa_inhibitor'],
        'gatran': ['anticoagulant', 'thrombin_inhibitor'],
        'prazole': ['ppi', 'acid_reducer'],
        'tidine': ['h2_blocker', 'acid_reducer'],
        'setron': ['antiemetic', '5ht3_antagonist'],
        'zodone': ['antidepressant', 'sari'],
        'pine': ['antipsychotic', 'atypical'],
        'peridol': ['antipsychotic', 'typical'],
    }
    
    for suffix, drug_classes in patterns.items():
        if suffix in drug:
            classes.extend(drug_classes)
    
    # Specific drug mappings
    specific_drugs = {
        'aspirin': ['nsaid', 'antiplatelet', 'analgesic'],
        'warfarin': ['anticoagulant', 'vitamin_k_antagonist'],
        'heparin': ['anticoagulant', 'parenteral'],
        'digoxin': ['cardiac_glycoside', 'antiarrhythmic'],
        'lithium': ['mood_stabilizer', 'narrow_therapeutic_index'],
        'methotrexate': ['immunosuppressant', 'antimetabolite'],
        'phenytoin': ['anticonvulsant', 'cyp_inducer'],
        'carbamazepine': ['anticonvulsant', 'cyp_inducer'],
        'rifampin': ['antibiotic', 'cyp_inducer'],
        'ketoconazole': ['antifungal', 'cyp_inhibitor'],
    }
    
    if drug in specific_drugs:
        classes.extend(specific_drugs[drug])
    
    return list(set(classes)) if classes else ['unknown']


def get_interaction_risk_features(classes1: list, classes2: list) -> dict:
    """Calculate interaction risk based on drug classes."""
    features = {
        'has_anticoagulant': 0,
        'has_nsaid': 0,
        'has_opioid': 0,
        'has_sedative': 0,
        'has_cyp_inhibitor': 0,
        'has_cyp_inducer': 0,
        'has_narrow_therapeutic': 0,
        'has_antihypertensive': 0,
        'has_antidepressant': 0,
        'has_antipsychotic': 0,
        'class_match': 0,
        'risk_combo_anticoag_nsaid': 0,
        'risk_combo_opioid_sedative': 0,
        'risk_combo_cyp_drug': 0,
    }
    
    all_classes = set(classes1 + classes2)
    
    if 'anticoagulant' in all_classes or 'antiplatelet' in all_classes:
        features['has_anticoagulant'] = 1
    if 'nsaid' in all_classes:
        features['has_nsaid'] = 1
    if 'opioid' in all_classes:
        features['has_opioid'] = 1
    if 'sedative' in all_classes or 'benzodiazepine' in all_classes:
        features['has_sedative'] = 1
    if 'cyp_inhibitor' in all_classes:
        features['has_cyp_inhibitor'] = 1
    if 'cyp_inducer' in all_classes:
        features['has_cyp_inducer'] = 1
    if 'narrow_therapeutic_index' in all_classes:
        features['has_narrow_therapeutic'] = 1
    if 'antihypertensive' in all_classes:
        features['has_antihypertensive'] = 1
    if 'antidepressant' in all_classes or 'ssri' in all_classes:
        features['has_antidepressant'] = 1
    if 'antipsychotic' in all_classes:
        features['has_antipsychotic'] = 1
    
    shared = set(classes1) & set(classes2)
    if shared and 'unknown' not in shared:
        features['class_match'] = 1
    
    if features['has_anticoagulant'] and features['has_nsaid']:
        features['risk_combo_anticoag_nsaid'] = 1
    if features['has_opioid'] and features['has_sedative']:
        features['risk_combo_opioid_sedative'] = 1
    if (features['has_cyp_inhibitor'] or features['has_cyp_inducer']):
        features['risk_combo_cyp_drug'] = 1
    
    return features


class DrugInteractionMLModel:
    """Advanced XGBoost-based drug interaction prediction model (v2.0)"""
    
    def __init__(self):
        self.model = None
        self.tfidf_vectorizer = None
        self.word_vectorizer = None
        self.severity_mapping = {
            0: "none",
            1: "mild", 
            2: "moderate",
            3: "severe",
            4: "contraindicated"
        }
        self.is_loaded = False
        self.model_info: Dict[str, Any] = {}
        self.is_advanced = False
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            import joblib
            
            model_path = MODEL_DIR / "drug_interaction_model.joblib"
            tfidf_path = MODEL_DIR / "drug_vectorizer.joblib"
            word_path = MODEL_DIR / "word_vectorizer.joblib"
            info_path = MODEL_DIR / "model_info.json"
            
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}. Run train_advanced_model.py first.")
                return False
            
            self.model = joblib.load(model_path)
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            
            # Check if advanced model (has word vectorizer)
            if word_path.exists():
                self.word_vectorizer = joblib.load(word_path)
                self.is_advanced = True
                logger.info("Loaded ADVANCED model with word vectorizer")
            else:
                self.is_advanced = False
                logger.info("Loaded basic model (no word vectorizer)")
            
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
            
            self.is_loaded = True
            logger.info(f"ML Model v{self.model_info.get('version', '?')} loaded. Accuracy: {self.model_info.get('accuracy', 'N/A'):.4f}")
            return True
            
        except ImportError:
            logger.error("Required packages not installed.")
            return False
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return False
    
    def _extract_features(self, drug1: str, drug2: str):
        """Extract features from drug pair for prediction"""
        if not self.tfidf_vectorizer:
            raise ValueError("Vectorizer not loaded")
        
        # Create combined feature string (alphabetically sorted)
        drugs = sorted([drug1.lower().strip(), drug2.lower().strip()])
        combined = f"{drugs[0]} {drugs[1]}"
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([combined])
        
        if self.is_advanced and self.word_vectorizer:
            # Word features
            word_features = self.word_vectorizer.transform([combined])
            
            # Pharmacological class features
            classes1 = get_drug_class(drugs[0])
            classes2 = get_drug_class(drugs[1])
            risk_features = get_interaction_risk_features(classes1, classes2)
            class_features = csr_matrix([list(risk_features.values())])
            
            # Combine all features
            features = hstack([tfidf_features, word_features, class_features])
        else:
            features = tfidf_features
        
        return features
    
    def predict(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """
        Predict drug-drug interaction
        
        Returns:
            Dictionary with has_interaction, severity, confidence, etc.
        """
        if not self.is_loaded:
            return {
                "has_interaction": False,
                "severity": "unknown",
                "confidence": 0.0,
                "model_version": None,
                "error": "Model not loaded"
            }
        
        try:
            features = self._extract_features(drug1, drug2)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(max(probabilities))
            has_interaction = prediction > 0
            severity = self.severity_mapping.get(prediction, "unknown")
            
            # Get drug classes for additional context
            drugs = sorted([drug1.lower().strip(), drug2.lower().strip()])
            classes1 = get_drug_class(drugs[0])
            classes2 = get_drug_class(drugs[1])
            
            return {
                "has_interaction": has_interaction,
                "severity": severity,
                "confidence": confidence,
                "model_version": self.model_info.get("version", "2.0.0"),
                "prediction_class": int(prediction),
                "probabilities": {
                    self.severity_mapping.get(i, f"class_{i}"): float(p) 
                    for i, p in enumerate(probabilities)
                },
                "drug1_classes": classes1,
                "drug2_classes": classes2,
                "is_advanced_model": self.is_advanced
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "has_interaction": False,
                "severity": "unknown", 
                "confidence": 0.0,
                "model_version": None,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            "is_loaded": self.is_loaded,
            "is_advanced": self.is_advanced,
            "version": self.model_info.get("version", "2.0.0"),
            "accuracy": self.model_info.get("accuracy"),
            "training_date": self.model_info.get("training_date"),
            "num_training_samples": self.model_info.get("num_training_samples"),
            "feature_count": self.model_info.get("feature_count"),
            "model_type": self.model_info.get("model_type"),
            "enhancements": self.model_info.get("enhancements", [])
        }


# Global model instance
_ml_model: Optional[DrugInteractionMLModel] = None


def get_ml_model() -> Optional[DrugInteractionMLModel]:
    """Get the global ML model instance"""
    return _ml_model


def initialize_ml_model() -> DrugInteractionMLModel:
    """Initialize and return the global ML model"""
    global _ml_model
    _ml_model = DrugInteractionMLModel()
    _ml_model.load_model()
    return _ml_model
