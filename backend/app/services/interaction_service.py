"""
Drug Interaction Service.

Core business logic for checking drug interactions and finding safe alternatives.
"""
from typing import List, Optional, Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func
from sqlalchemy.orm import selectinload
from difflib import SequenceMatcher
import logging

from app.models import Drug, DrugInteraction, DrugSimilarity, Category
from app.schemas import (
    InteractionCheckResponse, AlternativeDrug, AlternativeSuggestionResponse,
    DrugResponse, InteractionResponse, SeverityLevel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractionService:
    """Service for checking drug interactions and finding alternatives."""
    
    SEVERITY_RANKING = {
        "contraindicated": 4,
        "major": 3,
        "moderate": 2,
        "minor": 1
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def search_drugs(self, query: str, limit: int = 10) -> List[Drug]:
        """
        Search for drugs by name (supports partial matching).
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching drugs
        """
        query = query.strip().upper()
        
        # Search by name, generic name, or brand names
        stmt = select(Drug).where(
            or_(
                func.upper(Drug.name).contains(query),
                func.upper(Drug.generic_name).contains(query),
                func.upper(Drug.brand_names).contains(query)
            )
        ).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_drug_by_name(self, name: str) -> Optional[Drug]:
        """
        Get a drug by exact or fuzzy name match.
        
        Args:
            name: Drug name to search
            
        Returns:
            Drug object or None
        """
        name = name.strip().upper()
        
        # Try exact match first
        stmt = select(Drug).where(func.upper(Drug.name) == name)
        result = await self.db.execute(stmt)
        drug = result.scalar_one_or_none()
        
        if drug:
            return drug
        
        # Try generic name
        stmt = select(Drug).where(func.upper(Drug.generic_name) == name)
        result = await self.db.execute(stmt)
        drug = result.scalar_one_or_none()
        
        if drug:
            return drug
        
        # Try partial match
        stmt = select(Drug).where(
            or_(
                func.upper(Drug.name).contains(name),
                func.upper(Drug.generic_name).contains(name)
            )
        ).limit(1)
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def check_interaction(self, drug1_name: str, drug2_name: str) -> InteractionCheckResponse:
        """
        Check if two drugs have a known interaction.
        
        Args:
            drug1_name: First drug name
            drug2_name: Second drug name
            
        Returns:
            InteractionCheckResponse with interaction details
        """
        # Find both drugs
        drug1 = await self.get_drug_by_name(drug1_name)
        drug2 = await self.get_drug_by_name(drug2_name)
        
        if not drug1:
            # Create a temporary response for unknown drug
            return self._create_unknown_drug_response(drug1_name, "first")
        
        if not drug2:
            return self._create_unknown_drug_response(drug2_name, "second")
        
        # Check for interaction
        interaction = await self._get_interaction(drug1.id, drug2.id)
        
        # Build response
        drug1_response = DrugResponse.model_validate(drug1)
        drug2_response = DrugResponse.model_validate(drug2)
        
        if interaction:
            severity = interaction.severity
            is_safe = severity == "minor"
            
            safety_messages = {
                "contraindicated": "⛔ CONTRAINDICATED: These drugs should NOT be used together under any circumstances.",
                "major": "⚠️ MAJOR INTERACTION: These drugs have a significant interaction. Consult your healthcare provider immediately.",
                "moderate": "⚡ MODERATE INTERACTION: Use caution. Monitor for side effects and consult your pharmacist.",
                "minor": "ℹ️ MINOR INTERACTION: Generally safe but be aware of potential mild effects."
            }
            
            recommendations = self._get_recommendations(interaction)
            
            interaction_response = InteractionResponse(
                id=interaction.id,
                severity=SeverityLevel(interaction.severity),
                description=interaction.description,
                effect=interaction.effect,
                mechanism=interaction.mechanism,
                management=interaction.management,
                drug1=drug1_response,
                drug2=drug2_response,
                source=interaction.source,
                evidence_level=interaction.evidence_level,
                confidence_score=interaction.confidence_score,
                created_at=interaction.created_at
            )
            
            return InteractionCheckResponse(
                drug1=drug1_response,
                drug2=drug2_response,
                has_interaction=True,
                is_safe=is_safe,
                interaction=interaction_response,
                safety_message=safety_messages.get(severity, "Unknown interaction severity."),
                recommendations=recommendations
            )
        else:
            return InteractionCheckResponse(
                drug1=drug1_response,
                drug2=drug2_response,
                has_interaction=False,
                is_safe=True,
                interaction=None,
                safety_message="✅ NO KNOWN INTERACTION: These drugs appear to be safe to use together based on available data.",
                recommendations=[
                    "Always inform your healthcare provider of all medications you take.",
                    "Monitor for any unexpected side effects.",
                    "Absence of known interactions doesn't guarantee complete safety."
                ]
            )
    
    async def _get_interaction(self, drug1_id: int, drug2_id: int) -> Optional[DrugInteraction]:
        """Get interaction between two drugs (order-independent)."""
        stmt = select(DrugInteraction).where(
            or_(
                and_(
                    DrugInteraction.drug1_id == drug1_id,
                    DrugInteraction.drug2_id == drug2_id
                ),
                and_(
                    DrugInteraction.drug1_id == drug2_id,
                    DrugInteraction.drug2_id == drug1_id
                )
            )
        ).options(
            selectinload(DrugInteraction.drug1),
            selectinload(DrugInteraction.drug2)
        )
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    def _get_recommendations(self, interaction: DrugInteraction) -> List[str]:
        """Generate recommendations based on interaction."""
        recommendations = []
        
        if interaction.management:
            recommendations.append(f"Management: {interaction.management}")
        
        severity_recommendations = {
            "contraindicated": [
                "Do NOT take these medications together.",
                "Contact your doctor for alternative medications.",
                "Seek immediate medical advice if you have already taken both."
            ],
            "major": [
                "Consult your healthcare provider before continuing.",
                "Consider asking about alternative medications.",
                "Monitor closely for adverse effects."
            ],
            "moderate": [
                "Take medications at different times if possible.",
                "Monitor for increased side effects.",
                "Inform your pharmacist about both medications."
            ],
            "minor": [
                "Generally safe but remain aware of potential effects.",
                "Report any unusual symptoms to your healthcare provider."
            ]
        }
        
        recommendations.extend(severity_recommendations.get(interaction.severity, []))
        
        return recommendations
    
    def _create_unknown_drug_response(self, drug_name: str, position: str) -> InteractionCheckResponse:
        """Create response for unknown drug."""
        from datetime import datetime
        
        unknown_drug = DrugResponse(
            id=0,
            name=drug_name,
            is_approved=False,
            created_at=datetime.utcnow()
        )
        
        placeholder_drug = DrugResponse(
            id=0,
            name="Unknown",
            is_approved=False,
            created_at=datetime.utcnow()
        )
        
        if position == "first":
            drug1, drug2 = unknown_drug, placeholder_drug
        else:
            drug1, drug2 = placeholder_drug, unknown_drug
        
        return InteractionCheckResponse(
            drug1=drug1,
            drug2=drug2,
            has_interaction=False,
            is_safe=False,
            interaction=None,
            safety_message=f"⚠️ DRUG NOT FOUND: '{drug_name}' was not found in our database. Please verify the spelling or try an alternative name.",
            recommendations=[
                "Check the spelling of the drug name.",
                "Try using the generic name instead of brand name.",
                "Consult your pharmacist for verification."
            ]
        )
    
    async def find_alternatives(
        self,
        drug1_name: str,
        drug2_name: str,
        max_alternatives: int = 5
    ) -> AlternativeSuggestionResponse:
        """
        Find safe alternative drugs when an interaction is detected.
        
        Args:
            drug1_name: First drug with interaction
            drug2_name: Second drug with interaction
            max_alternatives: Maximum alternatives to suggest
            
        Returns:
            AlternativeSuggestionResponse with safe alternatives
        """
        drug1 = await self.get_drug_by_name(drug1_name)
        drug2 = await self.get_drug_by_name(drug2_name)
        
        if not drug1 or not drug2:
            raise ValueError("One or both drugs not found")
        
        # Find alternatives for drug1 (similar drugs that don't interact with drug2)
        alternatives_for_drug1 = await self._find_safe_alternatives(drug1, drug2, max_alternatives)
        
        # Find alternatives for drug2 (similar drugs that don't interact with drug1)
        alternatives_for_drug2 = await self._find_safe_alternatives(drug2, drug1, max_alternatives)
        
        # Find safe combinations
        safe_combinations = await self._find_safe_combinations(
            alternatives_for_drug1,
            alternatives_for_drug2
        )
        
        drug1_response = DrugResponse.model_validate(drug1)
        drug2_response = DrugResponse.model_validate(drug2)
        
        return AlternativeSuggestionResponse(
            original_drug1=drug1_response,
            original_drug2=drug2_response,
            alternatives_for_drug1=alternatives_for_drug1,
            alternatives_for_drug2=alternatives_for_drug2,
            safe_combinations=safe_combinations
        )
    
    async def _find_safe_alternatives(
        self,
        target_drug: Drug,
        other_drug: Drug,
        max_alternatives: int
    ) -> List[AlternativeDrug]:
        """Find similar drugs that don't interact with the other drug."""
        alternatives = []
        
        # Get drugs in the same class
        similar_drugs = await self._get_similar_drugs(target_drug, limit=20)
        
        for similar_drug, similarity_score in similar_drugs:
            # Skip the original drugs
            if similar_drug.id in (target_drug.id, other_drug.id):
                continue
            
            # Check if this alternative interacts with the other drug
            interaction = await self._get_interaction(similar_drug.id, other_drug.id)
            
            has_interaction = interaction is not None
            interaction_severity = interaction.severity if interaction else None
            
            # Only suggest if no interaction or minor interaction
            if not has_interaction or interaction_severity == "minor":
                drug_response = DrugResponse.model_validate(similar_drug)
                
                reason = f"Similar to {target_drug.name}"
                if similar_drug.drug_class:
                    reason += f" (Same class: {similar_drug.drug_class})"
                
                alternatives.append(AlternativeDrug(
                    drug=drug_response,
                    similarity_score=similarity_score,
                    reason=reason,
                    has_interaction_with_other=has_interaction
                ))
                
                if len(alternatives) >= max_alternatives:
                    break
        
        # Sort by similarity score and interaction status
        alternatives.sort(key=lambda x: (-int(not x.has_interaction_with_other), -x.similarity_score))
        
        return alternatives[:max_alternatives]
    
    async def _get_similar_drugs(
        self,
        drug: Drug,
        limit: int = 20
    ) -> List[Tuple[Drug, float]]:
        """Get drugs similar to the given drug."""
        similar_drugs = []
        
        # Get drugs in the same class
        if drug.drug_class:
            stmt = select(Drug).where(
                and_(
                    Drug.drug_class == drug.drug_class,
                    Drug.id != drug.id
                )
            ).limit(limit)
            
            result = await self.db.execute(stmt)
            class_drugs = result.scalars().all()
            
            for d in class_drugs:
                # Calculate similarity based on various factors
                similarity = self._calculate_drug_similarity(drug, d)
                similar_drugs.append((d, similarity))
        
        # Also check drug similarity table
        stmt = select(DrugSimilarity).where(
            or_(
                DrugSimilarity.drug1_id == drug.id,
                DrugSimilarity.drug2_id == drug.id
            )
        ).order_by(DrugSimilarity.overall_similarity.desc()).limit(limit)
        
        result = await self.db.execute(stmt)
        similarities = result.scalars().all()
        
        for sim in similarities:
            other_id = sim.drug2_id if sim.drug1_id == drug.id else sim.drug1_id
            
            # Get the drug
            stmt = select(Drug).where(Drug.id == other_id)
            result = await self.db.execute(stmt)
            other_drug = result.scalar_one_or_none()
            
            if other_drug:
                # Check if not already in list
                if not any(d[0].id == other_drug.id for d in similar_drugs):
                    similar_drugs.append((other_drug, sim.overall_similarity))
        
        # Sort by similarity
        similar_drugs.sort(key=lambda x: x[1], reverse=True)
        
        return similar_drugs[:limit]
    
    def _calculate_drug_similarity(self, drug1: Drug, drug2: Drug) -> float:
        """Calculate similarity score between two drugs."""
        score = 0.0
        weight_sum = 0.0
        
        # Same drug class (weight: 0.4)
        if drug1.drug_class and drug2.drug_class:
            if drug1.drug_class.upper() == drug2.drug_class.upper():
                score += 0.4
            weight_sum += 0.4
        
        # Similar indication (weight: 0.3)
        if drug1.indication and drug2.indication:
            indication_sim = SequenceMatcher(
                None,
                drug1.indication.lower()[:200],
                drug2.indication.lower()[:200]
            ).ratio()
            score += 0.3 * indication_sim
            weight_sum += 0.3
        
        # Similar mechanism (weight: 0.2)
        if drug1.mechanism and drug2.mechanism:
            mechanism_sim = SequenceMatcher(
                None,
                drug1.mechanism.lower()[:200],
                drug2.mechanism.lower()[:200]
            ).ratio()
            score += 0.2 * mechanism_sim
            weight_sum += 0.2
        
        # Name similarity (weight: 0.1)
        name_sim = SequenceMatcher(
            None,
            drug1.name.lower(),
            drug2.name.lower()
        ).ratio()
        score += 0.1 * name_sim
        weight_sum += 0.1
        
        # Normalize score
        if weight_sum > 0:
            return score / weight_sum
        return 0.0
    
    async def _find_safe_combinations(
        self,
        alternatives1: List[AlternativeDrug],
        alternatives2: List[AlternativeDrug]
    ) -> List[Dict]:
        """Find safe combinations from alternatives."""
        safe_combinations = []
        
        for alt1 in alternatives1:
            for alt2 in alternatives2:
                # Check interaction between alternatives
                interaction = await self._get_interaction(alt1.drug.id, alt2.drug.id)
                
                if not interaction or interaction.severity == "minor":
                    safe_combinations.append({
                        "drug1": {
                            "name": alt1.drug.name,
                            "id": alt1.drug.id
                        },
                        "drug2": {
                            "name": alt2.drug.name,
                            "id": alt2.drug.id
                        },
                        "combined_similarity": (alt1.similarity_score + alt2.similarity_score) / 2,
                        "interaction_status": "minor" if interaction else "none"
                    })
        
        # Sort by combined similarity
        safe_combinations.sort(key=lambda x: x["combined_similarity"], reverse=True)
        
        return safe_combinations[:10]  # Return top 10 combinations
    
    async def get_all_interactions_for_drug(
        self,
        drug_name: str,
        severity_filter: Optional[str] = None
    ) -> List[DrugInteraction]:
        """Get all known interactions for a drug."""
        drug = await self.get_drug_by_name(drug_name)
        
        if not drug:
            return []
        
        stmt = select(DrugInteraction).where(
            or_(
                DrugInteraction.drug1_id == drug.id,
                DrugInteraction.drug2_id == drug.id
            )
        ).options(
            selectinload(DrugInteraction.drug1),
            selectinload(DrugInteraction.drug2)
        )
        
        if severity_filter:
            stmt = stmt.where(DrugInteraction.severity == severity_filter)
        
        stmt = stmt.order_by(
            # Order by severity (most severe first)
            func.case(
                (DrugInteraction.severity == 'contraindicated', 1),
                (DrugInteraction.severity == 'major', 2),
                (DrugInteraction.severity == 'moderate', 3),
                (DrugInteraction.severity == 'minor', 4),
            )
        )
        
        result = await self.db.execute(stmt)
        return result.scalars().all()


def create_interaction_service(db: AsyncSession) -> InteractionService:
    """Factory function to create interaction service."""
    return InteractionService(db)

