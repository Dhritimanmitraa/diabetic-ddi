"""
Robust Drug Data Fetcher with multiple fallback sources.

This module provides production-grade drug information fetching
with automatic fallback between multiple data sources:
1. DailyMed (NLM) - Most reliable, free
2. RxNorm (NIH) - Always available
3. OpenFDA - Rate limited but comprehensive
4. PubChem - Chemical information
"""
import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.services.api_client import get_api_client, RobustAPIClient

logger = logging.getLogger(__name__)


@dataclass
class DrugInfo:
    """Standardized drug information from any source."""
    name: str
    generic_name: Optional[str] = None
    brand_names: List[str] = field(default_factory=list)
    drug_class: Optional[str] = None
    description: Optional[str] = None
    uses: Optional[str] = None
    mechanism: Optional[str] = None
    side_effects: Optional[str] = None
    warnings: Optional[str] = None
    dosage: Optional[str] = None
    interactions: List[str] = field(default_factory=list)
    source: str = "unknown"
    rxcui: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "generic_name": self.generic_name,
            "brand_names": self.brand_names,
            "drug_class": self.drug_class,
            "description": self.description,
            "uses": self.uses,
            "mechanism": self.mechanism,
            "side_effects": self.side_effects,
            "warnings": self.warnings,
            "dosage": self.dosage,
            "interactions": self.interactions,
            "source": self.source,
            "rxcui": self.rxcui,
            "fetched_at": self.fetched_at.isoformat(),
        }
    
    def to_text(self) -> str:
        """Convert to text format for RAG indexing."""
        parts = [f"Drug Name: {self.name}"]
        
        if self.generic_name:
            parts.append(f"Generic Name: {self.generic_name}")
        if self.drug_class:
            parts.append(f"Drug Class: {self.drug_class}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.uses:
            parts.append(f"Uses: {self.uses}")
        if self.mechanism:
            parts.append(f"Mechanism: {self.mechanism}")
        if self.side_effects:
            parts.append(f"Side Effects: {self.side_effects}")
        if self.warnings:
            parts.append(f"Warnings: {self.warnings}")
        if self.dosage:
            parts.append(f"Dosage: {self.dosage}")
        if self.interactions:
            parts.append(f"Known Interactions: {', '.join(self.interactions[:10])}")
            
        return "\n\n".join(parts)


@dataclass
class InteractionInfo:
    """Drug-drug interaction information."""
    drug1: str
    drug2: str
    severity: str  # minor, moderate, major, contraindicated
    description: str
    effect: Optional[str] = None
    management: Optional[str] = None
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug1": self.drug1,
            "drug2": self.drug2,
            "severity": self.severity,
            "description": self.description,
            "effect": self.effect,
            "management": self.management,
            "source": self.source,
        }


class RobustDrugFetcher:
    """
    Production-grade drug data fetcher with multiple fallback sources.
    
    Tries sources in order of reliability:
    1. DailyMed (NLM) - Most reliable, comprehensive drug labels
    2. RxNorm (NIH) - Standardized drug names, always available
    3. OpenFDA - FDA drug labels and adverse events
    4. PubChem - Chemical compound information
    """
    
    # API endpoints
    DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
    OPENFDA_BASE = "https://api.fda.gov/drug"
    PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def __init__(self, api_key_openfda: Optional[str] = None):
        self.api_key_openfda = api_key_openfda or os.getenv("OPENFDA_API_KEY", "")
        self.client = get_api_client()
    
    # ========== DailyMed (Primary Source) ==========
    
    async def fetch_from_dailymed(self, drug_name: str) -> Optional[DrugInfo]:
        """
        Fetch drug info from DailyMed (NLM).
        
        DailyMed is the most reliable source with comprehensive FDA-approved
        drug labels. It's free and has generous rate limits.
        """
        # Search for drug
        search_url = f"{self.DAILYMED_BASE}/spls.json"
        params = {"drug_name": drug_name, "pagesize": 1}
        
        data = await self.client.fetch(
            url=search_url,
            source="dailymed",
            params=params,
            cache_ttl=86400,  # 24 hours
        )
        
        if not data or not data.get("data"):
            return None
        
        spls = data.get("data", [])
        if not spls:
            return None
        
        spl = spls[0]
        
        # Extract information
        drug_info = DrugInfo(
            name=drug_name.title(),
            source="dailymed",
        )
        
        # Get title (usually brand name)
        title = spl.get("title", "")
        if title:
            drug_info.brand_names = [title.split("-")[0].strip()]
        
        # Get product info
        products = spl.get("products", [])
        if products:
            product = products[0]
            drug_info.generic_name = product.get("active_ingredients", [{}])[0].get("name")
            drug_info.dosage = product.get("dosage_form")
        
        # Get the full label for more details
        setid = spl.get("setid")
        if setid:
            label_data = await self._fetch_dailymed_label(setid)
            if label_data:
                drug_info.description = label_data.get("description")
                drug_info.uses = label_data.get("indications")
                drug_info.warnings = label_data.get("warnings")
                drug_info.side_effects = label_data.get("adverse_reactions")
                drug_info.mechanism = label_data.get("mechanism")
        
        logger.info(f"[OK] DailyMed: Found {drug_name}")
        return drug_info
    
    async def _fetch_dailymed_label(self, setid: str) -> Optional[Dict]:
        """Fetch detailed label sections from DailyMed."""
        url = f"{self.DAILYMED_BASE}/spls/{setid}.json"
        
        data = await self.client.fetch(url=url, source="dailymed")
        if not data:
            return None
        
        sections = {}
        for section in data.get("data", {}).get("sections", []):
            name = section.get("name", "").lower()
            text = section.get("text", "")[:1000]  # Limit text length
            
            if "indications" in name:
                sections["indications"] = text
            elif "description" in name:
                sections["description"] = text
            elif "warnings" in name or "precautions" in name:
                sections["warnings"] = text
            elif "adverse" in name:
                sections["adverse_reactions"] = text
            elif "mechanism" in name or "clinical pharmacology" in name:
                sections["mechanism"] = text
        
        return sections
    
    # ========== RxNorm (Always Available) ==========
    
    async def fetch_from_rxnorm(self, drug_name: str) -> Optional[DrugInfo]:
        """
        Fetch drug info from NIH RxNorm API.
        
        RxNorm provides standardized drug names and is always available.
        Good for getting RxCUI and drug class information.
        """
        # Search for drug
        search_url = f"{self.RXNORM_BASE}/drugs.json"
        params = {"name": drug_name}
        
        data = await self.client.fetch(
            url=search_url,
            source="rxnorm",
            params=params,
            cache_ttl=86400,
        )
        
        if not data:
            return None
        
        drug_group = data.get("drugGroup", {})
        concept_groups = drug_group.get("conceptGroup", [])
        
        if not concept_groups:
            return None
        
        drug_info = DrugInfo(
            name=drug_name.title(),
            source="rxnorm",
        )
        
        # Extract concept properties
        for group in concept_groups:
            props = group.get("conceptProperties", [])
            if props:
                prop = props[0]
                drug_info.rxcui = prop.get("rxcui")
                drug_info.generic_name = prop.get("name")
                
                # Get drug class using RxCUI
                if drug_info.rxcui:
                    drug_class = await self._fetch_rxnorm_class(drug_info.rxcui)
                    if drug_class:
                        drug_info.drug_class = drug_class
                break
        
        # Get interactions for this drug
        if drug_info.rxcui:
            interactions = await self._fetch_rxnorm_interactions(drug_info.rxcui)
            drug_info.interactions = interactions[:20]  # Limit
        
        logger.info(f"[OK] RxNorm: Found {drug_name} (RxCUI: {drug_info.rxcui})")
        return drug_info
    
    async def _fetch_rxnorm_class(self, rxcui: str) -> Optional[str]:
        """Fetch drug class from RxNorm."""
        url = f"{self.RXNORM_BASE}/rxclass/class/byRxcui.json"
        params = {"rxcui": rxcui}
        
        data = await self.client.fetch(url=url, source="rxnorm", params=params)
        if not data:
            return None
        
        classes = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
        if classes:
            return classes[0].get("rxclassMinConceptItem", {}).get("className")
        
        return None
    
    async def _fetch_rxnorm_interactions(self, rxcui: str) -> List[str]:
        """Fetch drug interactions from RxNorm."""
        url = f"{self.RXNORM_BASE}/interaction/interaction.json"
        params = {"rxcui": rxcui}
        
        data = await self.client.fetch(url=url, source="rxnorm", params=params)
        if not data:
            return []
        
        interactions = []
        groups = data.get("interactionTypeGroup", [])
        
        for group in groups:
            for itype in group.get("interactionType", []):
                for pair in itype.get("interactionPair", []):
                    desc = pair.get("description", "")
                    if desc:
                        interactions.append(desc[:200])
        
        return interactions
    
    # ========== OpenFDA (Comprehensive Labels) ==========
    
    async def fetch_from_openfda(self, drug_name: str) -> Optional[DrugInfo]:
        """
        Fetch drug info from OpenFDA.
        
        FDA database with comprehensive drug labels.
        Rate limited but has rich information.
        """
        url = f"{self.OPENFDA_BASE}/label.json"
        params = {
            "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            "limit": 1,
        }
        
        # Add API key if available
        if self.api_key_openfda:
            params["api_key"] = self.api_key_openfda
        
        data = await self.client.fetch(
            url=url,
            source="openfda",
            params=params,
            cache_ttl=86400,
        )
        
        if not data or not data.get("results"):
            return None
        
        result = data["results"][0]
        openfda = result.get("openfda", {})
        
        drug_info = DrugInfo(
            name=drug_name.title(),
            source="openfda",
            generic_name=openfda.get("generic_name", [None])[0],
            brand_names=openfda.get("brand_name", []),
            drug_class=openfda.get("pharm_class_epc", [None])[0] if openfda.get("pharm_class_epc") else None,
            description=result.get("description", [None])[0] if result.get("description") else None,
            uses=result.get("indications_and_usage", [None])[0] if result.get("indications_and_usage") else None,
            mechanism=result.get("mechanism_of_action", [None])[0] if result.get("mechanism_of_action") else None,
            warnings=result.get("warnings", [None])[0] if result.get("warnings") else None,
            side_effects=result.get("adverse_reactions", [None])[0] if result.get("adverse_reactions") else None,
            dosage=result.get("dosage_and_administration", [None])[0] if result.get("dosage_and_administration") else None,
        )
        
        # Truncate long fields
        if drug_info.description and len(drug_info.description) > 1000:
            drug_info.description = drug_info.description[:1000] + "..."
        if drug_info.uses and len(drug_info.uses) > 800:
            drug_info.uses = drug_info.uses[:800] + "..."
        
        logger.info(f"[OK] OpenFDA: Found {drug_name}")
        return drug_info
    
    # ========== PubChem (Chemical Info) ==========
    
    async def fetch_from_pubchem(self, drug_name: str) -> Optional[DrugInfo]:
        """
        Fetch drug info from PubChem.
        
        Chemical compound database with molecular information.
        Useful as a last resort fallback.
        """
        # Search for compound
        url = f"{self.PUBCHEM_BASE}/compound/name/{drug_name}/JSON"
        
        data = await self.client.fetch(
            url=url,
            source="pubchem",
            cache_ttl=86400,
        )
        
        if not data or not data.get("PC_Compounds"):
            return None
        
        compound = data["PC_Compounds"][0]
        
        drug_info = DrugInfo(
            name=drug_name.title(),
            source="pubchem",
        )
        
        # Extract compound ID
        compound_id = compound.get("id", {}).get("id", {}).get("cid")
        
        # Get more properties
        if compound_id:
            props = await self._fetch_pubchem_properties(compound_id)
            if props:
                drug_info.description = props.get("description")
        
        logger.info(f"[OK] PubChem: Found {drug_name}")
        return drug_info
    
    async def _fetch_pubchem_properties(self, cid: int) -> Optional[Dict]:
        """Fetch compound properties from PubChem."""
        url = f"{self.PUBCHEM_BASE}/compound/cid/{cid}/description/JSON"
        
        data = await self.client.fetch(url=url, source="pubchem")
        if not data:
            return None
        
        info_list = data.get("InformationList", {}).get("Information", [])
        if info_list:
            return {"description": info_list[0].get("Description", "")}
        
        return None
    
    # ========== Main Fetch Method (with Fallback) ==========
    
    async def fetch_drug_info(
        self,
        drug_name: str,
        sources: Optional[List[str]] = None,
    ) -> Optional[DrugInfo]:
        """
        Fetch drug information with automatic fallback.
        
        Tries multiple sources in order until one succeeds.
        
        Args:
            drug_name: Name of the drug to look up
            sources: Optional list of sources to try (default: all in order)
            
        Returns:
            DrugInfo if found, None otherwise
        """
        if not drug_name or len(drug_name) < 2:
            return None
        
        drug_name = drug_name.strip()
        
        # Default source order (most reliable first)
        if sources is None:
            sources = ["dailymed", "rxnorm", "openfda", "pubchem"]
        
        fetch_methods = {
            "dailymed": self.fetch_from_dailymed,
            "rxnorm": self.fetch_from_rxnorm,
            "openfda": self.fetch_from_openfda,
            "pubchem": self.fetch_from_pubchem,
        }
        
        for source in sources:
            if source not in fetch_methods:
                continue
                
            try:
                logger.info(f"Trying {source} for {drug_name}...")
                result = await fetch_methods[source](drug_name)
                
                if result:
                    return result
                    
            except Exception as e:
                logger.warning(f"{source} failed for {drug_name}: {e}")
                continue
        
        logger.warning(f"All sources failed for {drug_name}")
        return None
    
    async def fetch_interactions(
        self,
        drug1: str,
        drug2: str,
    ) -> List[InteractionInfo]:
        """
        Fetch drug-drug interactions from multiple sources.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            
        Returns:
            List of interaction information
        """
        interactions = []
        
        # Try RxNorm interactions
        drug1_info = await self.fetch_drug_info(drug1, sources=["rxnorm"])
        
        if drug1_info and drug1_info.rxcui:
            url = f"{self.RXNORM_BASE}/interaction/list.json"
            params = {"rxcuis": drug1_info.rxcui}
            
            data = await self.client.fetch(url=url, source="rxnorm", params=params)
            
            if data:
                groups = data.get("fullInteractionTypeGroup", [])
                for group in groups:
                    for itype in group.get("fullInteractionType", []):
                        for pair in itype.get("interactionPair", []):
                            concepts = pair.get("interactionConcept", [])
                            if len(concepts) >= 2:
                                name2 = concepts[1].get("minConceptItem", {}).get("name", "")
                                
                                if drug2.lower() in name2.lower():
                                    interactions.append(InteractionInfo(
                                        drug1=drug1,
                                        drug2=drug2,
                                        severity=pair.get("severity", "unknown"),
                                        description=pair.get("description", ""),
                                        source="rxnorm",
                                    ))
        
        return interactions
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all data sources."""
        return self.client.get_health_status()


# Singleton instance
_fetcher: Optional[RobustDrugFetcher] = None


def get_robust_fetcher() -> RobustDrugFetcher:
    """Get or create the singleton drug fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = RobustDrugFetcher()
    return _fetcher
