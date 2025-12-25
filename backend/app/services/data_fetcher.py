"""
Data fetching service for collecting drug-drug interaction data from public sources.

Sources:
1. OpenFDA - FDA Adverse Event Reporting System
2. DrugBank (Open Data)
3. NIH Drug Interaction API
4. PubChem
"""
import asyncio
import aiohttp
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.services.cache import cache_get_json, cache_set_json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugDataFetcher:
    """Fetches drug and interaction data from multiple public APIs."""
    
    # API Endpoints
    OPENFDA_BASE = "https://api.fda.gov/drug"
    NIH_DDI_BASE = "https://rxnav.nlm.nih.gov/REST"
    PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    async def fetch_openfda_drugs(self, limit: int = 10000) -> List[Dict]:
        """
        Fetch drug information from OpenFDA.
        
        The OpenFDA API provides access to FDA drug labels and adverse events.
        """
        cache_key = f"openfda_drugs:{limit}"
        cached = await cache_get_json(cache_key)
        if cached:
            logger.info("Returning OpenFDA drugs from cache")
            return cached

        drugs = []
        skip = 0
        batch_size = 1000
        
        async with aiohttp.ClientSession() as session:
            while len(drugs) < limit:
                url = f"{self.OPENFDA_BASE}/label.json"
                params = {
                    "limit": min(batch_size, limit - len(drugs)),
                    "skip": skip
                }
                
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get("results", [])
                            
                            if not results:
                                break
                                
                            for item in results:
                                drug_info = self._parse_openfda_drug(item)
                                if drug_info:
                                    drugs.append(drug_info)
                            
                            skip += batch_size
                            logger.info(f"Fetched {len(drugs)} drugs from OpenFDA...")
                            
                            # Rate limiting
                            await asyncio.sleep(0.5)
                        else:
                            logger.warning(f"OpenFDA API returned status {response.status}")
                            break
                            
                except Exception as e:
                    logger.error(f"Error fetching from OpenFDA: {e}")
                    break
        
        # Cache result for 6 hours
        await cache_set_json(cache_key, drugs, ttl_seconds=21600)
        return drugs
    
    def _parse_openfda_drug(self, item: Dict) -> Optional[Dict]:
        """Parse OpenFDA drug label into standardized format."""
        try:
            openfda = item.get("openfda", {})
            
            # Get drug names
            brand_names = openfda.get("brand_name", [])
            generic_names = openfda.get("generic_name", [])
            substance_names = openfda.get("substance_name", [])
            
            if not (brand_names or generic_names):
                return None
            
            return {
                "name": (brand_names[0] if brand_names else generic_names[0]).title(),
                "generic_name": generic_names[0].title() if generic_names else None,
                "brand_names": json.dumps(brand_names),
                "drug_class": openfda.get("pharm_class_epc", [""])[0] if openfda.get("pharm_class_epc") else None,
                "description": item.get("description", [""])[0] if item.get("description") else None,
                "indication": item.get("indications_and_usage", [""])[0] if item.get("indications_and_usage") else None,
                "mechanism": item.get("mechanism_of_action", [""])[0] if item.get("mechanism_of_action") else None,
                "drugbank_id": openfda.get("drugbank_id", [None])[0] if openfda.get("drugbank_id") else None,
            }
        except Exception as e:
            logger.debug(f"Error parsing drug: {e}")
            return None

    async def fetch_openfda_interactions(self, drug_names: List[str], limit: int = 100000) -> List[Dict]:
        """
        Fetch drug interactions from OpenFDA adverse events.
        
        Uses adverse event reports to identify drug combinations with reported issues.
        """
        cache_key = f"openfda_interactions:{limit}"
        cached = await cache_get_json(cache_key)
        if cached:
            logger.info("Returning OpenFDA interactions from cache")
            return cached
        interactions = []
        
        async with aiohttp.ClientSession() as session:
            # Fetch adverse events involving multiple drugs
            url = f"{self.OPENFDA_BASE}/event.json"
            
            skip = 0
            batch_size = 1000
            
            while len(interactions) < limit:
                params = {
                    "search": "patient.drug.drugcharacterization:1",  # Primary suspect drugs
                    "limit": min(batch_size, limit - len(interactions)),
                    "skip": skip
                }
                
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get("results", [])
                            
                            if not results:
                                break
                            
                            for event in results:
                                event_interactions = self._extract_interactions_from_event(event)
                                interactions.extend(event_interactions)
                            
                            skip += batch_size
                            logger.info(f"Processed {skip} adverse events, found {len(interactions)} potential interactions...")
                            
                            await asyncio.sleep(0.5)
                        else:
                            break
                            
                except Exception as e:
                    logger.error(f"Error fetching adverse events: {e}")
                    break
        
        interactions = interactions[:limit]
        await cache_set_json(cache_key, interactions, ttl_seconds=21600)
        return interactions
    
    def _extract_interactions_from_event(self, event: Dict) -> List[Dict]:
        """Extract drug-drug interactions from an adverse event report."""
        interactions = []
        
        try:
            drugs = event.get("patient", {}).get("drug", [])
            reactions = event.get("patient", {}).get("reaction", [])
            
            # Need at least 2 drugs for an interaction
            if len(drugs) < 2:
                return []
            
            # Extract drug names
            drug_names = []
            for drug in drugs:
                name = drug.get("medicinalproduct", "").strip()
                if name:
                    drug_names.append(name.upper())
            
            # Get reaction descriptions
            reaction_desc = ", ".join([r.get("reactionmeddrapt", "") for r in reactions[:3]])
            
            # Get severity
            seriousness = event.get("serious", 0)
            if seriousness:
                severity = "major"
            elif event.get("seriousnessother"):
                severity = "moderate"
            else:
                severity = "minor"
            
            # Create pairwise interactions
            for i in range(len(drug_names)):
                for j in range(i + 1, len(drug_names)):
                    interactions.append({
                        "drug1_name": drug_names[i],
                        "drug2_name": drug_names[j],
                        "severity": severity,
                        "effect": reaction_desc,
                        "source": "openfda",
                        "evidence_level": "case_report",
                        "confidence_score": 0.6
                    })
        
        except Exception as e:
            logger.debug(f"Error extracting interaction: {e}")
        
        return interactions

    async def fetch_rxnorm_interactions(self, rxcui: str) -> List[Dict]:
        """
        Fetch drug interactions from NIH RxNorm/RxNav API.
        
        RxNorm provides standardized drug names and interaction data.
        """
        interactions = []
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.NIH_DDI_BASE}/interaction/interaction.json"
            params = {"rxcui": rxcui}
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        interaction_groups = data.get("interactionTypeGroup", [])
                        
                        for group in interaction_groups:
                            for interaction_type in group.get("interactionType", []):
                                for pair in interaction_type.get("interactionPair", []):
                                    interaction_info = self._parse_rxnorm_interaction(pair)
                                    if interaction_info:
                                        interactions.append(interaction_info)
                                        
            except Exception as e:
                logger.error(f"Error fetching RxNorm interactions: {e}")
        
        return interactions
    
    def _parse_rxnorm_interaction(self, pair: Dict) -> Optional[Dict]:
        """Parse RxNorm interaction pair."""
        try:
            concepts = pair.get("interactionConcept", [])
            
            if len(concepts) < 2:
                return None
            
            drug1 = concepts[0].get("minConceptItem", {}).get("name", "")
            drug2 = concepts[1].get("minConceptItem", {}).get("name", "")
            
            description = pair.get("description", "")
            severity = pair.get("severity", "moderate").lower()
            
            # Map severity
            if "contraindicated" in severity.lower():
                severity = "contraindicated"
            elif "serious" in severity.lower() or "major" in severity.lower():
                severity = "major"
            elif "moderate" in severity.lower():
                severity = "moderate"
            else:
                severity = "minor"
            
            return {
                "drug1_name": drug1.upper(),
                "drug2_name": drug2.upper(),
                "severity": severity,
                "description": description,
                "source": "rxnorm",
                "evidence_level": "established",
                "confidence_score": 0.9
            }
            
        except Exception as e:
            logger.debug(f"Error parsing RxNorm interaction: {e}")
            return None

    async def fetch_all_rxnorm_drugs(self, limit: int = 5000) -> List[Dict]:
        """Fetch list of drugs from RxNorm."""
        cache_key = f"rxnorm_drugs:{limit}"
        cached = await cache_get_json(cache_key)
        if cached:
            logger.info("Returning RxNorm drugs from cache")
            return cached

        drugs = []
        
        async with aiohttp.ClientSession() as session:
            # Get all drug names
            url = f"{self.NIH_DDI_BASE}/allconcepts.json"
            params = {"tty": "IN"}  # Ingredient
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        concepts = data.get("minConceptGroup", {}).get("minConcept", [])
                        
                        for concept in concepts[:limit]:
                            drugs.append({
                                "rxcui": concept.get("rxcui"),
                                "name": concept.get("name", "").title(),
                                "tty": concept.get("tty")
                            })
                            
            except Exception as e:
                logger.error(f"Error fetching RxNorm drugs: {e}")
        
        await cache_set_json(cache_key, drugs, ttl_seconds=21600)
        return drugs

    async def fetch_comprehensive_data(self, 
                                        target_drugs: int = 5000,
                                        target_interactions: int = 100000) -> Dict:
        """
        Fetch comprehensive drug and interaction data from multiple sources.
        
        Returns a dictionary with drugs and interactions.
        """
        logger.info("Starting comprehensive data fetch...")
        
        # Fetch drugs from OpenFDA
        logger.info("Fetching drugs from OpenFDA...")
        openfda_drugs = await self.fetch_openfda_drugs(limit=target_drugs)
        
        # Fetch drugs from RxNorm
        logger.info("Fetching drugs from RxNorm...")
        rxnorm_drugs = await self.fetch_all_rxnorm_drugs(limit=target_drugs)
        
        # Merge and deduplicate drugs
        all_drugs = self._merge_drugs(openfda_drugs, rxnorm_drugs)
        logger.info(f"Total unique drugs: {len(all_drugs)}")
        
        # Fetch interactions
        logger.info("Fetching interactions from OpenFDA...")
        drug_names = [d["name"] for d in all_drugs]
        openfda_interactions = await self.fetch_openfda_interactions(drug_names, limit=target_interactions)
        
        # Fetch RxNorm interactions for top drugs
        logger.info("Fetching interactions from RxNorm...")
        rxnorm_interactions = []
        for drug in rxnorm_drugs[:500]:  # Top 500 drugs
            rxcui = drug.get("rxcui")
            if rxcui:
                interactions = await self.fetch_rxnorm_interactions(rxcui)
                rxnorm_interactions.extend(interactions)
                await asyncio.sleep(0.2)  # Rate limiting
        
        # Merge interactions
        all_interactions = self._merge_interactions(openfda_interactions, rxnorm_interactions)
        logger.info(f"Total unique interactions: {len(all_interactions)}")
        
        return {
            "drugs": all_drugs,
            "interactions": all_interactions,
            "metadata": {
                "fetched_at": datetime.utcnow().isoformat(),
                "sources": ["openfda", "rxnorm"],
                "drug_count": len(all_drugs),
                "interaction_count": len(all_interactions)
            }
        }
    
    def _merge_drugs(self, *drug_lists) -> List[Dict]:
        """Merge and deduplicate drug lists."""
        seen = set()
        merged = []
        
        for drug_list in drug_lists:
            for drug in drug_list:
                name_key = drug.get("name", "").upper().strip()
                if name_key and name_key not in seen:
                    seen.add(name_key)
                    merged.append(drug)
        
        return merged
    
    def _merge_interactions(self, *interaction_lists) -> List[Dict]:
        """Merge and deduplicate interaction lists."""
        seen = set()
        merged = []
        
        for interaction_list in interaction_lists:
            for interaction in interaction_list:
                # Create a unique key for the interaction pair
                d1 = interaction.get("drug1_name", "").upper()
                d2 = interaction.get("drug2_name", "").upper()
                key = tuple(sorted([d1, d2]))
                
                if key not in seen and d1 != d2:
                    seen.add(key)
                    merged.append(interaction)
        
        return merged

    def save_data(self, data: Dict, filename: str = "drug_data.json"):
        """Save fetched data to JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {filepath}")
        
    def load_data(self, filename: str = "drug_data.json") -> Optional[Dict]:
        """Load data from JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


async def main():
    """Main function to fetch and save drug data."""
    fetcher = DrugDataFetcher()
    
    # Fetch comprehensive data
    data = await fetcher.fetch_comprehensive_data(
        target_drugs=5000,
        target_interactions=100000
    )
    
    # Save to file
    fetcher.save_data(data)
    
    print(f"\nData collection complete!")
    print(f"Total drugs: {data['metadata']['drug_count']}")
    print(f"Total interactions: {data['metadata']['interaction_count']}")


if __name__ == "__main__":
    asyncio.run(main())

