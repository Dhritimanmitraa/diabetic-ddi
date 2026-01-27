"""
Drug-Food Interaction Service.

Provides lookup and analysis of interactions between medications and foods.
Critical for patient safety, especially for diabetic patients.

Features:
- Lookup by drug name (fuzzy matching)
- Severity-based filtering
- Food category explanations
- Timing recommendations
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FoodCategory:
    """Information about a food category."""
    id: str
    name: str
    description: str
    examples: List[str]


@dataclass
class FoodInteraction:
    """Single drug-food interaction."""
    drug_pattern: str
    drug_class: str
    food_category: FoodCategory
    severity: str  # contraindicated, major, moderate, minor
    effect: str
    recommendation: str
    timing: str  # ongoing, around_dose, with_dose


@dataclass
class FoodInteractionResult:
    """Result of a food interaction lookup."""
    drug_name: str
    drug_found: bool
    interactions: List[FoodInteraction]
    total_count: int
    has_contraindicated: bool
    has_major: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "drug_name": self.drug_name,
            "drug_found": self.drug_found,
            "total_interactions": self.total_count,
            "has_contraindicated": self.has_contraindicated,
            "has_major": self.has_major,
            "summary": self._generate_summary(),
            "interactions": [
                {
                    "food_category": {
                        "id": i.food_category.id,
                        "name": i.food_category.name,
                        "description": i.food_category.description,
                        "examples": i.food_category.examples
                    },
                    "severity": i.severity,
                    "effect": i.effect,
                    "recommendation": i.recommendation,
                    "timing": i.timing,
                    "timing_explanation": self._get_timing_explanation(i.timing)
                }
                for i in self.interactions
            ]
        }
    
    def _generate_summary(self) -> str:
        """Generate a patient-friendly summary."""
        if not self.interactions:
            return f"No significant food interactions found for {self.drug_name}."
        
        if self.has_contraindicated:
            return (
                f"⚠️ CRITICAL: {self.drug_name} has dangerous food interactions that "
                f"must be strictly avoided. Please review the details carefully."
            )
        elif self.has_major:
            return (
                f"⚠️ {self.drug_name} has {self.total_count} significant food interaction(s). "
                f"Please follow the dietary recommendations."
            )
        else:
            return (
                f"{self.drug_name} has {self.total_count} food consideration(s) to be aware of."
            )
    
    def _get_timing_explanation(self, timing: str) -> str:
        """Get human-readable timing explanation."""
        explanations = {
            "ongoing": "Avoid throughout treatment period",
            "around_dose": "Separate from medication by 2+ hours",
            "with_dose": "Relevant when taking medication"
        }
        return explanations.get(timing, timing)


# =============================================================================
# Food Interaction Service
# =============================================================================

class FoodInteractionService:
    """
    Service for looking up drug-food interactions.
    
    Loads data from JSON database and provides fuzzy matching
    for drug name lookup.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the service.
        
        Args:
            data_path: Path to food_interactions.json, auto-detected if None
        """
        self.data_path = data_path or self._find_data_file()
        self.data: Dict[str, Any] = {}
        self.food_categories: Dict[str, FoodCategory] = {}
        self.interactions: List[Dict[str, Any]] = []
        self.is_loaded = False
        
        self._load_data()
    
    def _find_data_file(self) -> str:
        """Find the food interactions data file."""
        # Try multiple locations
        possible_paths = [
            Path(__file__).parent.parent / "data" / "food_interactions.json",
            Path(__file__).parent / "data" / "food_interactions.json",
            Path("./app/data/food_interactions.json"),
            Path("./data/food_interactions.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # Return default path even if not found
        return str(possible_paths[0])
    
    def _load_data(self) -> bool:
        """Load food interaction data from JSON file."""
        try:
            if not os.path.exists(self.data_path):
                logger.warning(f"Food interaction data not found at {self.data_path}")
                return False
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # Parse food categories
            for cat_id, cat_data in self.data.get("food_categories", {}).items():
                self.food_categories[cat_id] = FoodCategory(
                    id=cat_id,
                    name=cat_data["name"],
                    description=cat_data["description"],
                    examples=cat_data["examples"]
                )
            
            # Store raw interactions
            self.interactions = self.data.get("interactions", [])
            
            self.is_loaded = True
            logger.info(
                f"Loaded {len(self.interactions)} food interactions "
                f"across {len(self.food_categories)} categories"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error loading food interactions: {e}")
            return False
    
    def get_interactions(
        self,
        drug_name: str,
        min_severity: Optional[str] = None,
        fuzzy_threshold: int = 70
    ) -> FoodInteractionResult:
        """
        Get food interactions for a drug.
        
        Args:
            drug_name: Name of the drug to look up
            min_severity: Minimum severity to include (contraindicated, major, moderate, minor)
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            
        Returns:
            FoodInteractionResult with all matching interactions
        """
        if not self.is_loaded:
            return FoodInteractionResult(
                drug_name=drug_name,
                drug_found=False,
                interactions=[],
                total_count=0,
                has_contraindicated=False,
                has_major=False
            )
        
        normalized_drug = drug_name.lower().strip()
        matched_interactions: List[FoodInteraction] = []
        
        severity_order = ["contraindicated", "major", "moderate", "minor"]
        min_severity_idx = severity_order.index(min_severity) if min_severity in severity_order else 3
        
        for interaction_data in self.interactions:
            drug_pattern = interaction_data["drug_pattern"].lower()
            
            # Check for match (exact, partial, or fuzzy)
            is_match = (
                normalized_drug == drug_pattern or
                normalized_drug in drug_pattern or
                drug_pattern in normalized_drug or
                fuzz.ratio(normalized_drug, drug_pattern) >= fuzzy_threshold
            )
            
            if is_match:
                severity = interaction_data["severity"]
                severity_idx = severity_order.index(severity) if severity in severity_order else 3
                
                # Filter by minimum severity
                if severity_idx <= min_severity_idx:
                    food_cat = self.food_categories.get(interaction_data["food_category"])
                    
                    if food_cat:
                        matched_interactions.append(FoodInteraction(
                            drug_pattern=interaction_data["drug_pattern"],
                            drug_class=interaction_data.get("drug_class", "unknown"),
                            food_category=food_cat,
                            severity=severity,
                            effect=interaction_data["effect"],
                            recommendation=interaction_data["recommendation"],
                            timing=interaction_data.get("timing", "ongoing")
                        ))
        
        # Sort by severity (most severe first)
        matched_interactions.sort(
            key=lambda x: severity_order.index(x.severity) if x.severity in severity_order else 10
        )
        
        return FoodInteractionResult(
            drug_name=drug_name,
            drug_found=len(matched_interactions) > 0,
            interactions=matched_interactions,
            total_count=len(matched_interactions),
            has_contraindicated=any(i.severity == "contraindicated" for i in matched_interactions),
            has_major=any(i.severity == "major" for i in matched_interactions)
        )
    
    def get_all_food_categories(self) -> List[Dict[str, Any]]:
        """Get all food categories with descriptions."""
        return [
            {
                "id": cat.id,
                "name": cat.name,
                "description": cat.description,
                "examples": cat.examples
            }
            for cat in self.food_categories.values()
        ]
    
    def get_drugs_by_food_category(self, category_id: str) -> List[Dict[str, Any]]:
        """
        Get all drugs that interact with a specific food category.
        
        Args:
            category_id: Food category ID (e.g., "grapefruit", "tyramine")
            
        Returns:
            List of drugs with their interaction details
        """
        if category_id not in self.food_categories:
            return []
        
        drugs = []
        for interaction in self.interactions:
            if interaction["food_category"] == category_id:
                drugs.append({
                    "drug": interaction["drug_pattern"],
                    "drug_class": interaction.get("drug_class", "unknown"),
                    "severity": interaction["severity"],
                    "effect": interaction["effect"],
                    "recommendation": interaction["recommendation"]
                })
        
        # Sort by severity
        severity_order = ["contraindicated", "major", "moderate", "minor"]
        drugs.sort(key=lambda x: severity_order.index(x["severity"]) if x["severity"] in severity_order else 10)
        
        return drugs
    
    def check_patient_medications(
        self,
        medications: List[str]
    ) -> Dict[str, Any]:
        """
        Check all food interactions for a patient's medication list.
        
        Args:
            medications: List of drug names the patient is taking
            
        Returns:
            Comprehensive food interaction report for all medications
        """
        all_interactions = []
        drugs_with_issues = []
        
        for med in medications:
            result = self.get_interactions(med)
            if result.interactions:
                drugs_with_issues.append(med)
                for interaction in result.interactions:
                    all_interactions.append({
                        "drug": med,
                        "food_category": interaction.food_category.name,
                        "severity": interaction.severity,
                        "effect": interaction.effect,
                        "recommendation": interaction.recommendation
                    })
        
        # Sort by severity
        severity_order = {"contraindicated": 0, "major": 1, "moderate": 2, "minor": 3}
        all_interactions.sort(key=lambda x: severity_order.get(x["severity"], 10))
        
        # Get unique food categories to avoid
        foods_to_avoid = {}
        for interaction in all_interactions:
            cat = interaction["food_category"]
            if cat not in foods_to_avoid or severity_order.get(interaction["severity"], 10) < severity_order.get(foods_to_avoid[cat]["severity"], 10):
                foods_to_avoid[cat] = interaction
        
        return {
            "total_medications_checked": len(medications),
            "medications_with_food_interactions": drugs_with_issues,
            "total_interactions_found": len(all_interactions),
            "has_critical_interactions": any(i["severity"] == "contraindicated" for i in all_interactions),
            "foods_to_avoid": list(foods_to_avoid.keys()),
            "all_interactions": all_interactions,
            "summary": self._generate_patient_summary(all_interactions, drugs_with_issues)
        }
    
    def _generate_patient_summary(
        self,
        interactions: List[Dict[str, Any]],
        affected_drugs: List[str]
    ) -> str:
        """Generate patient-friendly summary of all food interactions."""
        if not interactions:
            return "No significant food-drug interactions found for your medications. However, always follow your doctor's dietary advice."
        
        critical = [i for i in interactions if i["severity"] == "contraindicated"]
        major = [i for i in interactions if i["severity"] == "major"]
        
        if critical:
            return (
                f"⚠️ CRITICAL: You have {len(critical)} food restriction(s) that MUST be strictly followed "
                f"to avoid dangerous reactions. Please review the complete list carefully and discuss with your doctor."
            )
        elif major:
            return (
                f"You have {len(major)} significant food consideration(s) for your medications. "
                f"Following these guidelines will help your medications work properly and safely."
            )
        else:
            return (
                f"You have {len(interactions)} mild food consideration(s). "
                f"While not critical, following these recommendations can improve medication effectiveness."
            )


# =============================================================================
# Module-level singleton
# =============================================================================

_food_service: Optional[FoodInteractionService] = None


def get_food_interaction_service() -> FoodInteractionService:
    """Get or create the global food interaction service."""
    global _food_service
    
    if _food_service is None:
        _food_service = FoodInteractionService()
    
    return _food_service
