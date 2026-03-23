"""
Drug Interaction Service.

Core business logic for checking drug interactions and finding safe alternatives.
"""
from typing import List, Optional, Dict, Tuple
from itertools import combinations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func, case
from sqlalchemy.orm import selectinload
from difflib import SequenceMatcher
import json
import asyncio
import logging

from app.models import Drug, DrugInteraction, DrugSimilarity
from app.services.gemini_client import get_gemini_client
from app.schemas import (
    InteractionCheckResponse, AlternativeDrug, AlternativeSuggestionResponse,
    DrugResponse, InteractionResponse, SeverityLevel
)
from app.services.cache import interaction_cache, drug_lookup_cache

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

        cached = drug_lookup_cache.get(f"drug:{name}")
        if cached is not None:
            return cached

        # Try exact match first
        stmt = select(Drug).where(func.upper(Drug.name) == name)
        result = await self.db.execute(stmt)
        drug = result.scalar_one_or_none()
        
        if not drug:
            # Try generic name
            stmt = select(Drug).where(func.upper(Drug.generic_name) == name)
            result = await self.db.execute(stmt)
            drug = result.scalar_one_or_none()
        
        if not drug:
            # Try partial match
            stmt = select(Drug).where(
                or_(
                    func.upper(Drug.name).contains(name),
                    func.upper(Drug.generic_name).contains(name)
                )
            ).limit(1)
            result = await self.db.execute(stmt)
            drug = result.scalar_one_or_none()

        if drug is not None:
            drug_lookup_cache.set(f"drug:{name}", drug)
        return drug
    
    async def check_interaction(self, drug1_name: str, drug2_name: str) -> InteractionCheckResponse:
        """
        Check if two drugs have a known interaction.
        
        Args:
            drug1_name: First drug name
            drug2_name: Second drug name
            
        Returns:
            InteractionCheckResponse with interaction details
        """
        pair_key = f"ix:{min(drug1_name.upper(), drug2_name.upper())}:{max(drug1_name.upper(), drug2_name.upper())}"
        cached = interaction_cache.get(pair_key)
        if cached is not None:
            return cached

        drug1 = await self.get_drug_by_name(drug1_name)
        drug2 = await self.get_drug_by_name(drug2_name)
        
        if not drug1:
            return self._create_unknown_drug_response(drug1_name, "first")
        
        if not drug2:
            return self._create_unknown_drug_response(drug2_name, "second")
        
        interaction = await self._get_interaction(drug1.id, drug2.id)
        if not interaction:
            interaction = await self._check_interaction_via_llm(drug1, drug2)
            
        response = self._build_interaction_response(drug1, drug2, interaction)
        interaction_cache.set(pair_key, response)
        return response

    async def resolve_drugs_by_names(self, drug_names: List[str]) -> Dict[str, Optional[Drug]]:
        """Resolve a list of drug names while minimizing exact-match queries."""
        normalized = {
            name: name.strip().upper()
            for name in dict.fromkeys(drug_names)
            if name and name.strip()
        }
        exact_names = list({value for value in normalized.values() if value})
        resolved_by_upper: Dict[str, Drug] = {}

        if exact_names:
            stmt = select(Drug).where(
                or_(
                    func.upper(Drug.name).in_(exact_names),
                    func.upper(Drug.generic_name).in_(exact_names),
                )
            )
            result = await self.db.execute(stmt)
            for drug in result.scalars().all():
                if drug.name:
                    resolved_by_upper.setdefault(drug.name.strip().upper(), drug)
                if drug.generic_name:
                    resolved_by_upper.setdefault(drug.generic_name.strip().upper(), drug)

        resolved: Dict[str, Optional[Drug]] = {}
        for original_name, normalized_name in normalized.items():
            drug = resolved_by_upper.get(normalized_name)
            if drug is None:
                drug = await self.get_drug_by_name(original_name)
            resolved[original_name] = drug
        return resolved

    async def check_batch_interactions(self, drug_names: List[str]) -> List[Tuple[str, str, InteractionCheckResponse]]:
        """Check all unique drug pairs using a single interaction lookup query."""
        unique_names = [name.strip() for name in dict.fromkeys(drug_names) if name and name.strip()]
        resolved = await self.resolve_drugs_by_names(unique_names)
        resolved_ids = {
            name: drug.id
            for name, drug in resolved.items()
            if drug is not None
        }

        pair_conditions = []
        for drug1_name, drug2_name in combinations(unique_names, 2):
            drug1_id = resolved_ids.get(drug1_name)
            drug2_id = resolved_ids.get(drug2_name)
            if drug1_id and drug2_id:
                pair_conditions.append(
                    and_(DrugInteraction.drug1_id == drug1_id, DrugInteraction.drug2_id == drug2_id)
                )
                pair_conditions.append(
                    and_(DrugInteraction.drug1_id == drug2_id, DrugInteraction.drug2_id == drug1_id)
                )

        interaction_map: Dict[Tuple[int, int], DrugInteraction] = {}
        if pair_conditions:
            result = await self.db.execute(
                select(DrugInteraction)
                .where(or_(*pair_conditions))
                .options(
                    selectinload(DrugInteraction.drug1),
                    selectinload(DrugInteraction.drug2),
                )
            )
            for interaction in result.scalars().all():
                interaction_map[(interaction.drug1_id, interaction.drug2_id)] = interaction
                interaction_map[(interaction.drug2_id, interaction.drug1_id)] = interaction

        # Augment missing interactions dynamically via Gemini
        for drug1_name, drug2_name in combinations(unique_names, 2):
            drug1 = resolved.get(drug1_name)
            drug2 = resolved.get(drug2_name)
            if drug1 and drug2:
                if (drug1.id, drug2.id) not in interaction_map:
                    llm_interaction = await self._check_interaction_via_llm(drug1, drug2)
                    if llm_interaction:
                        interaction_map[(drug1.id, drug2.id)] = llm_interaction
                        interaction_map[(drug2.id, drug1.id)] = llm_interaction

        responses: List[Tuple[str, str, InteractionCheckResponse]] = []
        for drug1_name, drug2_name in combinations(unique_names, 2):
            drug1 = resolved.get(drug1_name)
            drug2 = resolved.get(drug2_name)

            if not drug1:
                response = self._create_unknown_drug_response(drug1_name, "first")
            elif not drug2:
                response = self._create_unknown_drug_response(drug2_name, "second")
            else:
                response = self._build_interaction_response(
                    drug1,
                    drug2,
                    interaction_map.get((drug1.id, drug2.id)),
                )
            responses.append((drug1_name, drug2_name, response))

        return responses
    
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
    
    async def _check_interaction_via_llm(self, drug1: Drug, drug2: Drug) -> Optional[DrugInteraction]:
        """Dynamically check for drug interactions using Gemini API when local DB misses."""
        client = get_gemini_client()
        if not client.is_available:
            return None

        prompt = f"""
        Analyze the potential drug-drug interaction between {drug1.name} and {drug2.name}.
        Return ONLY a JSON object with the following structure, and NO markdown code blocks or other text.
        If there is no clinically significant interaction, set severity to "none".

        {{
            "severity": "minor" | "moderate" | "major" | "contraindicated" | "none",
            "description": "Short description of the interaction",
            "effect": "Clinical effect of the interaction",
            "mechanism": "Pharmacological mechanism",
            "management": "How to manage this combination"
        }}
        """
        try:
            # Run the synchronous Gemini client in a thread pool to avoid blocking the event loop
            response = await asyncio.to_thread(client.generate_text, prompt, temperature=0.1)
            text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            
            if data.get("severity", "none").lower() == "none":
                return None
                
            severity = data.get("severity", "moderate").lower()
            if severity not in ["minor", "moderate", "major", "contraindicated"]:
                severity = "moderate"
                
            return DrugInteraction(
                id=99990000 + drug1.id + drug2.id,
                drug1_id=drug1.id,
                drug2_id=drug2.id,
                severity=severity,
                description=data.get("description", f"Potential interaction identified by AI between {drug1.name} and {drug2.name}."),
                effect=data.get("effect", ""),
                mechanism=data.get("mechanism", ""),
                management=data.get("management", ""),
                source="Gemini API",
                evidence_level="AI Generated",
                confidence_score=0.85,
                drug1=drug1,
                drug2=drug2
            )
        except Exception as e:
            logger.error(f"Error checking interaction via LLM: {e}")
            return None
    
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

    def _build_interaction_response(
        self,
        drug1: Drug,
        drug2: Drug,
        interaction: Optional[DrugInteraction],
    ) -> InteractionCheckResponse:
        """Build the response payload for a resolved drug pair."""
        drug1_response = DrugResponse.model_validate(drug1)
        drug2_response = DrugResponse.model_validate(drug2)

        if interaction:
            severity = interaction.severity
            is_safe = severity == "minor"

            safety_messages = {
                "contraindicated": "CONTRAINDICATED: These drugs should NOT be used together under any circumstances.",
                "major": "MAJOR INTERACTION: These drugs have a significant interaction. Consult your healthcare provider immediately.",
                "moderate": "MODERATE INTERACTION: Use caution. Monitor for side effects and consult your pharmacist.",
                "minor": "MINOR INTERACTION: Generally safe but be aware of potential mild effects.",
            }

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
                created_at=interaction.created_at,
            )

            return InteractionCheckResponse(
                drug1=drug1_response,
                drug2=drug2_response,
                has_interaction=True,
                is_safe=is_safe,
                interaction=interaction_response,
                safety_message=safety_messages.get(severity, "Unknown interaction severity."),
                recommendations=self._get_recommendations(interaction),
            )

        return InteractionCheckResponse(
            drug1=drug1_response,
            drug2=drug2_response,
            has_interaction=False,
            is_safe=True,
            interaction=None,
            safety_message="NO KNOWN INTERACTION: These drugs appear to be safe to use together based on available data.",
            recommendations=[
                "Always inform your healthcare provider of all medications you take.",
                "Monitor for any unexpected side effects.",
                "Absence of known interactions doesn't guarantee complete safety.",
            ],
        )
    
    def _create_unknown_drug_response(self, drug_name: str, position: str) -> InteractionCheckResponse:
        """Create response for unknown drug."""
        from datetime import datetime, timezone
        
        unknown_drug = DrugResponse(
            id=0,
            name=drug_name,
            is_approved=False,
            created_at=datetime.now(timezone.utc)
        )
        
        placeholder_drug = DrugResponse(
            id=0,
            name="Unknown",
            is_approved=False,
            created_at=datetime.now(timezone.utc)
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
            safety_message=f"DRUG NOT FOUND: '{drug_name}' was not found in our database. Please verify the spelling or try an alternative name.",
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
        candidate_ids = [
            similar_drug.id
            for similar_drug, _ in similar_drugs
            if similar_drug.id not in (target_drug.id, other_drug.id)
        ]
        interactions_by_candidate: Dict[int, DrugInteraction] = {}

        if candidate_ids:
            result = await self.db.execute(
                select(DrugInteraction).where(
                    or_(
                        and_(
                            DrugInteraction.drug1_id == other_drug.id,
                            DrugInteraction.drug2_id.in_(candidate_ids),
                        ),
                        and_(
                            DrugInteraction.drug2_id == other_drug.id,
                            DrugInteraction.drug1_id.in_(candidate_ids),
                        ),
                    )
                )
            )
            for interaction in result.scalars().all():
                candidate_id = (
                    interaction.drug2_id
                    if interaction.drug1_id == other_drug.id
                    else interaction.drug1_id
                )
                interactions_by_candidate[candidate_id] = interaction
        
        for similar_drug, similarity_score in similar_drugs:
            # Skip the original drugs
            if similar_drug.id in (target_drug.id, other_drug.id):
                continue
            
            interaction = interactions_by_candidate.get(similar_drug.id)
            
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
        
        # Batch fetch all related drugs (fix N+1 query pattern)
        other_ids = []
        for sim in similarities:
            other_id = sim.drug2_id if sim.drug1_id == drug.id else sim.drug1_id
            other_ids.append(other_id)
        
        if other_ids:
            # Single query to fetch all related drugs
            stmt = select(Drug).where(Drug.id.in_(other_ids))
            result = await self.db.execute(stmt)
            drugs_by_id = {d.id: d for d in result.scalars().all()}
            
            for sim in similarities:
                other_id = sim.drug2_id if sim.drug1_id == drug.id else sim.drug1_id
                other_drug = drugs_by_id.get(other_id)
                
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
        """Find safe combinations from alternatives using batch query."""
        if not alternatives1 or not alternatives2:
            return []

        pairs = [
            (alt1.drug.id, alt2.drug.id)
            for alt1 in alternatives1
            for alt2 in alternatives2
        ]

        all_ids = list({id for pair in pairs for id in pair})
        conditions = []
        for d1_id, d2_id in pairs:
            conditions.append(
                and_(DrugInteraction.drug1_id == d1_id, DrugInteraction.drug2_id == d2_id)
            )
            conditions.append(
                and_(DrugInteraction.drug1_id == d2_id, DrugInteraction.drug2_id == d1_id)
            )

        interactions_map = {}
        if conditions:
            stmt = select(DrugInteraction).where(or_(*conditions))
            result = await self.db.execute(stmt)
            for inter in result.scalars().all():
                key1 = (inter.drug1_id, inter.drug2_id)
                key2 = (inter.drug2_id, inter.drug1_id)
                interactions_map[key1] = inter
                interactions_map[key2] = inter

        safe_combinations = []
        for alt1 in alternatives1:
            for alt2 in alternatives2:
                interaction = interactions_map.get((alt1.drug.id, alt2.drug.id))
                if not interaction or interaction.severity == "minor":
                    safe_combinations.append({
                        "drug1": {"name": alt1.drug.name, "id": alt1.drug.id},
                        "drug2": {"name": alt2.drug.name, "id": alt2.drug.id},
                        "combined_similarity": (alt1.similarity_score + alt2.similarity_score) / 2,
                        "interaction_status": "minor" if interaction else "none"
                    })
        
        safe_combinations.sort(key=lambda x: x["combined_similarity"], reverse=True)
        return safe_combinations[:10]
    
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
            case(
                (DrugInteraction.severity == 'contraindicated', 1),
                (DrugInteraction.severity == 'major', 2),
                (DrugInteraction.severity == 'moderate', 3),
                (DrugInteraction.severity == 'minor', 4),
                else_=5,
            )
        )
        
        result = await self.db.execute(stmt)
        return result.scalars().all()


def create_interaction_service(db: AsyncSession) -> InteractionService:
    """Factory function to create interaction service."""
    return InteractionService(db)
