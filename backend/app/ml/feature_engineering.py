"""
Feature Engineering for Drug-Drug Interaction Prediction.

Extracts numerical features from drug pairs for machine learning models.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class DrugFeatureExtractor:
    """
    Extract features from drug pairs for ML prediction.

    Features include:
    - Drug class encoding (categorical)
    - Text similarity (mechanism, indication)
    - Interaction frequency statistics
    - Molecular properties (if available)
    """

    def __init__(self):
        self.drug_class_encoder = LabelEncoder()
        self.mechanism_vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english"
        )
        self.indication_vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english"
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.drug_class_map = {}
        self.interaction_freq = {}

    def fit(self, drugs: List[Dict], interactions: List[Dict]):
        """
        Fit the feature extractors on the drug dataset.

        Args:
            drugs: List of drug dictionaries with properties
            interactions: List of known interactions for frequency calculation
        """
        logger.info("Fitting feature extractor...")

        # Extract drug classes
        drug_classes = []
        for drug in drugs:
            drug_class = drug.get("drug_class") or "Unknown"
            drug_classes.append(drug_class)
            self.drug_class_map[drug.get("name", "").upper()] = drug_class

        # Fit drug class encoder
        unique_classes = list(set(drug_classes))
        if unique_classes:
            self.drug_class_encoder.fit(unique_classes + ["Unknown"])

        # Fit text vectorizers
        mechanisms = [drug.get("mechanism", "") or "" for drug in drugs]
        indications = [drug.get("indication", "") or "" for drug in drugs]

        # Filter empty strings
        mechanisms = [m if m else "no mechanism data" for m in mechanisms]
        indications = [i if i else "no indication data" for i in indications]

        self.mechanism_vectorizer.fit(mechanisms)
        self.indication_vectorizer.fit(indications)

        # Calculate interaction frequency per drug
        for interaction in interactions:
            drug1 = interaction.get("drug1_name", "").upper()
            drug2 = interaction.get("drug2_name", "").upper()

            self.interaction_freq[drug1] = self.interaction_freq.get(drug1, 0) + 1
            self.interaction_freq[drug2] = self.interaction_freq.get(drug2, 0) + 1

        self.is_fitted = True
        logger.info(
            f"Feature extractor fitted on {len(drugs)} drugs and {len(interactions)} interactions"
        )

    def extract_features(self, drug1: Dict, drug2: Dict) -> np.ndarray:
        """
        Extract feature vector for a drug pair.

        Args:
            drug1: First drug dictionary
            drug2: Second drug dictionary

        Returns:
            Feature vector as numpy array
        """
        if not self.is_fitted:
            raise ValueError(
                "Feature extractor must be fitted before extracting features"
            )

        features = []

        # 1. Drug class features (one-hot encoded difference)
        class1 = drug1.get("drug_class") or "Unknown"
        class2 = drug2.get("drug_class") or "Unknown"

        try:
            class1_encoded = self.drug_class_encoder.transform([class1])[0]
            class2_encoded = self.drug_class_encoder.transform([class2])[0]
        except ValueError:
            class1_encoded = self.drug_class_encoder.transform(["Unknown"])[0]
            class2_encoded = self.drug_class_encoder.transform(["Unknown"])[0]

        features.append(class1_encoded)
        features.append(class2_encoded)
        features.append(1 if class1 == class2 else 0)  # Same class indicator

        # 2. Mechanism similarity
        mech1 = drug1.get("mechanism", "") or "no mechanism data"
        mech2 = drug2.get("mechanism", "") or "no mechanism data"

        mech1_vec = self.mechanism_vectorizer.transform([mech1])
        mech2_vec = self.mechanism_vectorizer.transform([mech2])
        mech_similarity = cosine_similarity(mech1_vec, mech2_vec)[0][0]
        features.append(mech_similarity)

        # 3. Indication similarity
        ind1 = drug1.get("indication", "") or "no indication data"
        ind2 = drug2.get("indication", "") or "no indication data"

        ind1_vec = self.indication_vectorizer.transform([ind1])
        ind2_vec = self.indication_vectorizer.transform([ind2])
        ind_similarity = cosine_similarity(ind1_vec, ind2_vec)[0][0]
        features.append(ind_similarity)

        # 4. Interaction frequency features
        name1 = drug1.get("name", "").upper()
        name2 = drug2.get("name", "").upper()

        freq1 = self.interaction_freq.get(name1, 0)
        freq2 = self.interaction_freq.get(name2, 0)

        features.append(freq1)
        features.append(freq2)
        features.append(freq1 + freq2)  # Combined frequency
        features.append(abs(freq1 - freq2))  # Frequency difference

        # 5. Molecular weight features (if available)
        mw1 = drug1.get("molecular_weight") or 0
        mw2 = drug2.get("molecular_weight") or 0

        features.append(mw1)
        features.append(mw2)
        features.append(abs(mw1 - mw2) if mw1 and mw2 else 0)  # MW difference

        # 6. Text length features (proxy for complexity)
        desc1_len = len(drug1.get("description", "") or "")
        desc2_len = len(drug2.get("description", "") or "")

        features.append(np.log1p(desc1_len))
        features.append(np.log1p(desc2_len))

        # 7. Approval status
        features.append(1 if drug1.get("is_approved", True) else 0)
        features.append(1 if drug2.get("is_approved", True) else 0)

        return np.array(features, dtype=np.float32)

    def extract_batch_features(self, drug_pairs: List[Tuple[Dict, Dict]]) -> np.ndarray:
        """
        Extract features for multiple drug pairs.

        Args:
            drug_pairs: List of (drug1, drug2) tuples

        Returns:
            Feature matrix (n_pairs x n_features)
        """
        features = []
        for drug1, drug2 in drug_pairs:
            feat = self.extract_features(drug1, drug2)
            features.append(feat)

        return np.array(features)

    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        return [
            "drug1_class_encoded",
            "drug2_class_encoded",
            "same_class",
            "mechanism_similarity",
            "indication_similarity",
            "drug1_interaction_freq",
            "drug2_interaction_freq",
            "combined_interaction_freq",
            "interaction_freq_diff",
            "drug1_molecular_weight",
            "drug2_molecular_weight",
            "molecular_weight_diff",
            "drug1_description_length",
            "drug2_description_length",
            "drug1_approved",
            "drug2_approved",
        ]

    def save(self, filepath: str):
        """Save the fitted feature extractor."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "drug_class_encoder": self.drug_class_encoder,
                    "mechanism_vectorizer": self.mechanism_vectorizer,
                    "indication_vectorizer": self.indication_vectorizer,
                    "scaler": self.scaler,
                    "drug_class_map": self.drug_class_map,
                    "interaction_freq": self.interaction_freq,
                    "is_fitted": self.is_fitted,
                },
                f,
            )
        logger.info(f"Feature extractor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DrugFeatureExtractor":
        """Load a saved feature extractor."""
        extractor = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        extractor.drug_class_encoder = data["drug_class_encoder"]
        extractor.mechanism_vectorizer = data["mechanism_vectorizer"]
        extractor.indication_vectorizer = data["indication_vectorizer"]
        extractor.scaler = data["scaler"]
        extractor.drug_class_map = data["drug_class_map"]
        extractor.interaction_freq = data["interaction_freq"]
        extractor.is_fitted = data["is_fitted"]

        logger.info(f"Feature extractor loaded from {filepath}")
        return extractor


def prepare_training_data(
    drugs: List[Dict], interactions: List[Dict], negative_ratio: float = 1.0
) -> Tuple[List[Tuple[Dict, Dict]], List[int]]:
    """
    Prepare training data with positive and negative samples.

    Args:
        drugs: List of all drugs
        interactions: List of known interactions
        negative_ratio: Ratio of negative to positive samples

    Returns:
        Tuple of (drug_pairs, labels)
    """
    import random

    # Create drug lookup
    drug_lookup = {d.get("name", "").upper(): d for d in drugs}

    # Positive samples (known interactions)
    positive_pairs = []
    for interaction in interactions:
        drug1_name = interaction.get("drug1_name", "").upper()
        drug2_name = interaction.get("drug2_name", "").upper()

        drug1 = drug_lookup.get(drug1_name)
        drug2 = drug_lookup.get(drug2_name)

        if drug1 and drug2:
            positive_pairs.append((drug1, drug2))

    # Create set of known interaction pairs for quick lookup
    known_pairs = set()
    for interaction in interactions:
        d1 = interaction.get("drug1_name", "").upper()
        d2 = interaction.get("drug2_name", "").upper()
        known_pairs.add((d1, d2))
        known_pairs.add((d2, d1))

    # Negative samples (random pairs without known interactions)
    drug_names = list(drug_lookup.keys())
    num_negative = int(len(positive_pairs) * negative_ratio)
    negative_pairs = []

    attempts = 0
    max_attempts = num_negative * 10

    while len(negative_pairs) < num_negative and attempts < max_attempts:
        d1_name = random.choice(drug_names)
        d2_name = random.choice(drug_names)

        if d1_name != d2_name and (d1_name, d2_name) not in known_pairs:
            drug1 = drug_lookup[d1_name]
            drug2 = drug_lookup[d2_name]
            negative_pairs.append((drug1, drug2))
            known_pairs.add((d1_name, d2_name))  # Avoid duplicates

        attempts += 1

    # Combine and create labels
    all_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    # Shuffle
    combined = list(zip(all_pairs, labels))
    random.shuffle(combined)
    all_pairs, labels = zip(*combined)

    logger.info(
        f"Prepared {len(positive_pairs)} positive and {len(negative_pairs)} negative samples"
    )

    return list(all_pairs), list(labels)
