"""
Dynamic Drug Knowledge Base with Web Scraping.

Fetches drug information from trusted medical websites and caches in ChromaDB.
"""
import logging
import os
import json
import hashlib
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import httpx
from bs4 import BeautifulSoup

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Trusted medical information sources
MEDICAL_URLS = {
    # Drugs.com - comprehensive drug database
    "drugs_com": "https://www.drugs.com/{drug_name}.html",
    # MedlinePlus - NIH medical encyclopedia  
    "medlineplus": "https://medlineplus.gov/druginfo/meds/a{drug_id}.html",
    # RxList - drug information
    "rxlist": "https://www.rxlist.com/{drug_name}-drug.htm",
}

# Common drug name to URL mappings for quick lookup
DRUG_URL_MAPPINGS = {
    "paracetamol": "https://www.drugs.com/paracetamol.html",
    "acetaminophen": "https://www.drugs.com/acetaminophen.html",
    "ibuprofen": "https://www.drugs.com/ibuprofen.html",
    "aspirin": "https://www.drugs.com/aspirin.html",
    "metformin": "https://www.drugs.com/metformin.html",
    "amoxicillin": "https://www.drugs.com/amoxicillin.html",
    "azithromycin": "https://www.drugs.com/azithromycin.html",
    "omeprazole": "https://www.drugs.com/omeprazole.html",
    "pantoprazole": "https://www.drugs.com/pantoprazole.html",
    "atorvastatin": "https://www.drugs.com/atorvastatin.html",
    "amlodipine": "https://www.drugs.com/amlodipine.html",
    "lisinopril": "https://www.drugs.com/lisinopril.html",
    "losartan": "https://www.drugs.com/losartan.html",
    "metoprolol": "https://www.drugs.com/metoprolol.html",
    "prednisone": "https://www.drugs.com/prednisone.html",
    "cetirizine": "https://www.drugs.com/cetirizine.html",
    "loratadine": "https://www.drugs.com/loratadine.html",
    "sertraline": "https://www.drugs.com/sertraline.html",
    "fluoxetine": "https://www.drugs.com/fluoxetine.html",
    "gabapentin": "https://www.drugs.com/gabapentin.html",
    "tramadol": "https://www.drugs.com/tramadol.html",
    "ciprofloxacin": "https://www.drugs.com/ciprofloxacin.html",
    "doxycycline": "https://www.drugs.com/doxycycline.html",
    "levothyroxine": "https://www.drugs.com/levothyroxine.html",
    "warfarin": "https://www.drugs.com/warfarin.html",
    "clopidogrel": "https://www.drugs.com/clopidogrel.html",
    "furosemide": "https://www.drugs.com/furosemide.html",
    "hydrochlorothiazide": "https://www.drugs.com/hydrochlorothiazide.html",
    "albuterol": "https://www.drugs.com/albuterol.html",
    "salbutamol": "https://www.drugs.com/salbutamol.html",
    "montelukast": "https://www.drugs.com/montelukast.html",
    "alprazolam": "https://www.drugs.com/alprazolam.html",
    "diazepam": "https://www.drugs.com/diazepam.html",
    "insulin": "https://www.drugs.com/insulin.html",
    "glimepiride": "https://www.drugs.com/glimepiride.html",
    "rosuvastatin": "https://www.drugs.com/rosuvastatin.html",
    "escitalopram": "https://www.drugs.com/escitalopram.html",
    "pregabalin": "https://www.drugs.com/pregabalin.html",
    "domperidone": "https://www.drugs.com/international/domperidone.html",
    "ranitidine": "https://www.drugs.com/ranitidine.html",
    "ondansetron": "https://www.drugs.com/ondansetron.html",
    "loperamide": "https://www.drugs.com/loperamide.html",
    "lactulose": "https://www.drugs.com/lactulose.html",
    "colchicine": "https://www.drugs.com/colchicine.html",
    "allopurinol": "https://www.drugs.com/allopurinol.html",
    "vitamin d": "https://www.drugs.com/vitamin_d3.html",
    "vitamin b12": "https://www.drugs.com/vitamin_b12.html",
    "folic acid": "https://www.drugs.com/folic_acid.html",
    "iron": "https://www.drugs.com/ferrous_sulfate.html",
    "calcium": "https://www.drugs.com/calcium_carbonate.html",
}


class WebDrugKnowledgeBase:
    """
    Dynamic knowledge base that fetches drug info from the web.
    
    - Searches trusted medical websites for drug information
    - Caches results in ChromaDB for fast retrieval
    - Falls back to static knowledge if web fetch fails
    """
    
    COLLECTION_NAME = "web_drug_knowledge"
    CACHE_EXPIRY_DAYS = 7  # Re-fetch after 7 days
    
    def __init__(self):
        """Initialize the web knowledge base."""
        self.chroma_client = None
        self.embedding_function = None
        self.collection = None
        self.http_client = None
        self._init_chroma()
        self._init_http_client()
        self._load_static_knowledge()
    
    def _init_chroma(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            persist_dir = settings.CHROMA_PERSIST_DIR
            os.makedirs(persist_dir, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            from chromadb.utils import embedding_functions
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"type": "web_drug_knowledge"}
            )
            
            logger.info(f"Web Drug Knowledge Base initialized with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"ChromaDB initialization error: {e}")
    
    def _init_http_client(self):
        """Initialize HTTP client for web requests."""
        self.http_client = httpx.AsyncClient(
            timeout=15.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            follow_redirects=True
        )
    
    def _load_static_knowledge(self):
        """Load static drug knowledge as fallback."""
        self.static_drugs = {}
        try:
            knowledge_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data/drug_knowledge.json"
            )
            if os.path.exists(knowledge_path):
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for drug in data.get('drugs', []):
                        name = drug.get('name', '').lower()
                        self.static_drugs[name] = drug
                        # Also index by generic name
                        generic = drug.get('generic_name', '').lower()
                        if generic:
                            self.static_drugs[generic] = drug
                logger.info(f"Loaded {len(self.static_drugs)} static drugs as fallback")
        except Exception as e:
            logger.error(f"Error loading static knowledge: {e}")
    
    async def fetch_drug_info_from_web(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch drug information from trusted medical websites.
        
        Args:
            drug_name: Name of the drug to look up
            
        Returns:
            Dict with drug information if found
        """
        drug_name_lower = drug_name.lower().strip()
        
        # Check if we have a direct URL mapping
        url = DRUG_URL_MAPPINGS.get(drug_name_lower)
        
        if not url:
            # Try constructing URL from drug name
            drug_slug = drug_name_lower.replace(' ', '-').replace('_', '-')
            url = f"https://www.drugs.com/{drug_slug}.html"
        
        try:
            logger.info(f"Fetching drug info from: {url}")
            response = await self.http_client.get(url)
            
            if response.status_code == 200:
                return self._parse_drugs_com_page(response.text, drug_name)
            else:
                # Try alternative URL format
                alt_url = f"https://www.drugs.com/mtm/{drug_name_lower.replace(' ', '_')}.html"
                response = await self.http_client.get(alt_url)
                if response.status_code == 200:
                    return self._parse_drugs_com_page(response.text, drug_name)
                    
        except Exception as e:
            logger.warning(f"Error fetching from web for {drug_name}: {e}")
        
        return None
    
    def _parse_drugs_com_page(self, html: str, drug_name: str) -> Optional[Dict[str, Any]]:
        """Parse drug information from Drugs.com HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            drug_info = {
                "name": drug_name,
                "source": "drugs.com",
                "fetched_at": datetime.now().isoformat()
            }
            
            # Get generic name from title or header
            title = soup.find('h1')
            if title:
                drug_info['title'] = title.get_text(strip=True)
            
            # Look for drug description/overview
            overview = soup.find('div', class_='contentBox') or soup.find('section', class_='drug-overview')
            if overview:
                drug_info['overview'] = overview.get_text(strip=True)[:1500]
            
            # Look for "What is" section
            what_is = soup.find('h2', string=lambda t: t and 'what is' in t.lower())
            if what_is:
                next_p = what_is.find_next('p')
                if next_p:
                    drug_info['description'] = next_p.get_text(strip=True)
            
            # Look for uses section
            uses_header = soup.find(['h2', 'h3'], string=lambda t: t and ('uses' in t.lower() or 'used for' in t.lower()))
            if uses_header:
                uses_content = uses_header.find_next(['p', 'ul'])
                if uses_content:
                    drug_info['uses'] = uses_content.get_text(strip=True)[:800]
            
            # Look for side effects
            side_effects_header = soup.find(['h2', 'h3'], string=lambda t: t and 'side effects' in t.lower())
            if side_effects_header:
                se_content = side_effects_header.find_next(['p', 'ul'])
                if se_content:
                    drug_info['side_effects'] = se_content.get_text(strip=True)[:800]
            
            # Look for warnings
            warnings_header = soup.find(['h2', 'h3'], string=lambda t: t and 'warning' in t.lower())
            if warnings_header:
                warn_content = warnings_header.find_next(['p', 'ul'])
                if warn_content:
                    drug_info['warnings'] = warn_content.get_text(strip=True)[:500]
            
            # Build full text content for embedding
            full_text = f"""
Drug Name: {drug_info.get('title', drug_name)}
Description: {drug_info.get('description', drug_info.get('overview', 'N/A'))}
Uses: {drug_info.get('uses', 'See full prescribing information')}
Side Effects: {drug_info.get('side_effects', 'See full prescribing information')}
Warnings: {drug_info.get('warnings', 'Consult healthcare provider')}
Source: {drug_info.get('source', 'drugs.com')}
""".strip()
            
            drug_info['full_text'] = full_text
            
            return drug_info
            
        except Exception as e:
            logger.error(f"Error parsing drugs.com page: {e}")
            return None
    
    async def lookup_drug(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Look up drug information, checking cache first then web.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Drug information dict
        """
        drug_name_lower = drug_name.lower().strip()
        
        # 1. Check ChromaDB cache first
        cached = self._get_from_cache(drug_name_lower)
        if cached:
            logger.info(f"Found {drug_name} in cache")
            return cached
        
        # 2. Try to fetch from web
        web_info = await self.fetch_drug_info_from_web(drug_name)
        if web_info:
            # Cache the result
            self._add_to_cache(drug_name_lower, web_info)
            logger.info(f"Fetched and cached {drug_name} from web")
            return web_info
        
        # 3. Fall back to static knowledge
        static_info = self.static_drugs.get(drug_name_lower)
        if static_info:
            logger.info(f"Using static knowledge for {drug_name}")
            return {
                "name": static_info.get('name'),
                "full_text": f"""
Drug Name: {static_info.get('name')}
Generic Name: {static_info.get('generic_name', 'N/A')}
Drug Class: {static_info.get('drug_class', 'N/A')}
Uses: {static_info.get('uses', 'N/A')}
How It Works: {static_info.get('mechanism', 'N/A')}
Side Effects: {static_info.get('side_effects', 'N/A')}
Dosage: {static_info.get('dosage', 'N/A')}
Precautions: {static_info.get('precautions', 'N/A')}
""".strip(),
                "source": "static",
                **static_info
            }
        
        return None
    
    def _get_from_cache(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Get drug info from ChromaDB cache."""
        if not self.collection:
            return None
        
        try:
            # Search for exact match
            results = self.collection.query(
                query_texts=[drug_name],
                n_results=1,
                include=["documents", "metadatas"]
            )
            
            if results and results['documents'] and results['documents'][0]:
                metadata = results['metadatas'][0][0] if results['metadatas'] else {}
                
                # Check cache expiry
                fetched_at = metadata.get('fetched_at', '')
                if fetched_at:
                    try:
                        fetched_date = datetime.fromisoformat(fetched_at)
                        if datetime.now() - fetched_date > timedelta(days=self.CACHE_EXPIRY_DAYS):
                            logger.info(f"Cache expired for {drug_name}")
                            return None
                    except ValueError:
                        pass
                
                return {
                    "name": metadata.get('name', drug_name),
                    "full_text": results['documents'][0][0],
                    "source": metadata.get('source', 'cache'),
                    "cached": True
                }
                
        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
        
        return None
    
    def _add_to_cache(self, drug_name: str, drug_info: Dict[str, Any]):
        """Add drug info to ChromaDB cache."""
        if not self.collection:
            return
        
        try:
            doc_id = f"web_{hashlib.md5(drug_name.encode()).hexdigest()}"
            
            self.collection.upsert(
                documents=[drug_info.get('full_text', str(drug_info))],
                metadatas=[{
                    "name": drug_info.get('name', drug_name),
                    "source": drug_info.get('source', 'web'),
                    "fetched_at": datetime.now().isoformat()
                }],
                ids=[doc_id]
            )
            
            logger.info(f"Cached drug info for {drug_name}")
            
        except Exception as e:
            logger.error(f"Cache add error: {e}")
    
    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            List of matching documents
        """
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            matches = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    matches.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0
                    })
            
            return matches
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []


# Singleton instance
_web_knowledge_base: Optional[WebDrugKnowledgeBase] = None


def get_web_drug_knowledge_base() -> WebDrugKnowledgeBase:
    """Get or create the web drug knowledge base singleton."""
    global _web_knowledge_base
    if _web_knowledge_base is None:
        _web_knowledge_base = WebDrugKnowledgeBase()
    return _web_knowledge_base
