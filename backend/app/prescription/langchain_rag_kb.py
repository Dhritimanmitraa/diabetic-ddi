"""
LangChain RAG Knowledge Base.

Implements IBM-style RAG using:
- WebBaseLoader to fetch content from medical URLs
- RecursiveCharacterTextSplitter for chunking
- ChromaDB for vector storage
- Retriever for semantic search
"""
import logging
import os
from typing import Optional, List, Dict, Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Medical knowledge URLs - trusted sources for drug information
MEDICAL_KNOWLEDGE_URLS = [
    # Drugs.com - comprehensive drug database
    "https://www.drugs.com/drug_information.html",
    "https://www.drugs.com/drug-classes.html",
    # MedlinePlus - NIH medical encyclopedia
    "https://medlineplus.gov/druginformation.html",
    # FDA Drug Information  
    "https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-glossary-terms",
    # WebMD Drug Database
    "https://www.webmd.com/drugs/2/index",
    # RxList - Drug Index
    "https://www.rxlist.com/drugs/alpha_a.htm",
]

# Pre-defined drug URLs for common medications
DRUG_SPECIFIC_URLS = {
    "metformin": [
        "https://www.drugs.com/metformin.html",
        "https://medlineplus.gov/druginfo/meds/a696005.html",
    ],
    "ibuprofen": [
        "https://www.drugs.com/ibuprofen.html",
        "https://medlineplus.gov/druginfo/meds/a682159.html",
    ],
    "paracetamol": [
        "https://www.drugs.com/paracetamol.html",
        "https://www.drugs.com/acetaminophen.html",
    ],
    "aspirin": [
        "https://www.drugs.com/aspirin.html",
        "https://medlineplus.gov/druginfo/meds/a682878.html",
    ],
    "amoxicillin": [
        "https://www.drugs.com/amoxicillin.html",
        "https://medlineplus.gov/druginfo/meds/a685001.html",
    ],
    "omeprazole": [
        "https://www.drugs.com/omeprazole.html",
        "https://medlineplus.gov/druginfo/meds/a693050.html",
    ],
    "atorvastatin": [
        "https://www.drugs.com/atorvastatin.html",
        "https://medlineplus.gov/druginfo/meds/a600045.html",
    ],
    "amlodipine": [
        "https://www.drugs.com/amlodipine.html",
        "https://medlineplus.gov/druginfo/meds/a692044.html",
    ],
    "lisinopril": [
        "https://www.drugs.com/lisinopril.html",
        "https://medlineplus.gov/druginfo/meds/a692051.html",
    ],
    "losartan": [
        "https://www.drugs.com/losartan.html",
        "https://medlineplus.gov/druginfo/meds/a695008.html",
    ],
}


class LangChainRAGKnowledgeBase:
    """
    IBM-style LangChain RAG implementation.
    
    Uses WebBaseLoader to fetch content from medical URLs,
    chunks it with RecursiveCharacterTextSplitter,
    and stores in ChromaDB for retrieval.
    """
    
    COLLECTION_NAME = "medical_knowledge_rag"
    
    def __init__(self):
        """Initialize the RAG knowledge base."""
        self.vectorstore = None
        self.retriever = None
        self.embeddings = None
        self._init_embeddings()
        self._init_vectorstore()
    
    def _init_embeddings(self):
        """Initialize embeddings using sentence-transformers."""
        try:
            from chromadb.utils import embedding_functions
            self.embeddings = embedding_functions.DefaultEmbeddingFunction()
            logger.info("Embeddings initialized (sentence-transformers)")
        except Exception as e:
            logger.error(f"Embeddings init error: {e}")
    
    def _init_vectorstore(self):
        """Initialize ChromaDB vector store."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            persist_dir = settings.CHROMA_PERSIST_DIR
            os.makedirs(persist_dir, exist_ok=True)
            
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            self.vectorstore = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embeddings,
                metadata={"type": "medical_rag"}
            )
            
            logger.info(f"RAG VectorStore initialized with {self.vectorstore.count()} documents")
            
            # Index static knowledge if empty
            if self.vectorstore.count() == 0:
                self._index_static_knowledge()
                
        except Exception as e:
            logger.error(f"VectorStore init error: {e}")
    
    def _index_static_knowledge(self):
        """Index static drug knowledge from JSON file."""
        try:
            import json
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            knowledge_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data/drug_knowledge.json"
            )
            
            if not os.path.exists(knowledge_path):
                logger.warning("Static knowledge file not found")
                return
            
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            drugs = data.get('drugs', [])
            
            # Create text splitter following IBM pattern
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            documents = []
            metadatas = []
            ids = []
            
            for i, drug in enumerate(drugs):
                # Create comprehensive drug document
                drug_text = f"""
Drug Name: {drug.get('name', 'Unknown')}
Generic Name: {drug.get('generic_name', 'N/A')}
Drug Class: {drug.get('drug_class', 'N/A')}

What is {drug.get('name', 'this drug')} used for?
{drug.get('uses', 'Information not available.')}

How does {drug.get('name', 'this drug')} work?
{drug.get('mechanism', 'Mechanism not specified.')}

What are the side effects of {drug.get('name', 'this drug')}?
{drug.get('side_effects', 'Consult your doctor for side effect information.')}

Dosage Information:
{drug.get('dosage', 'Follow your doctor\'s prescription.')}

Precautions and Warnings:
{drug.get('precautions', 'Consult your healthcare provider.')}
""".strip()
                
                # Split into chunks
                chunks = text_splitter.split_text(drug_text)
                
                for j, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "drug_name": drug.get('name', 'Unknown'),
                        "generic_name": drug.get('generic_name', ''),
                        "drug_class": drug.get('drug_class', ''),
                        "chunk_index": j,
                        "source": "static_knowledge"
                    })
                    ids.append(f"drug_{i}_{j}_{drug.get('name', 'unknown').lower().replace(' ', '_')}")
            
            # Add to vectorstore
            if documents:
                self.vectorstore.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Indexed {len(documents)} chunks from {len(drugs)} drugs")
                
        except Exception as e:
            logger.error(f"Error indexing static knowledge: {e}")
    
    async def load_url_content(self, url: str) -> List[str]:
        """
        Load content from a URL using LangChain WebBaseLoader pattern.
        
        Args:
            url: URL to load content from
            
        Returns:
            List of document chunks
        """
        try:
            from langchain_community.document_loaders import WebBaseLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            # Load the document
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            if not docs:
                return []
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            
            chunks = []
            for doc in docs:
                splits = text_splitter.split_text(doc.page_content)
                chunks.extend(splits)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading URL {url}: {e}")
            return []
    
    async def index_drug_from_urls(self, drug_name: str) -> bool:
        """
        Fetch and index drug information from web URLs.
        
        Args:
            drug_name: Name of the drug to look up
            
        Returns:
            True if successfully indexed
        """
        drug_name_lower = drug_name.lower().strip()
        
        # Get URLs for this drug
        urls = DRUG_SPECIFIC_URLS.get(drug_name_lower, [])
        
        if not urls:
            # Try constructing URL
            urls = [f"https://www.drugs.com/{drug_name_lower}.html"]
        
        all_chunks = []
        
        for url in urls:
            chunks = await self.load_url_content(url)
            all_chunks.extend(chunks)
        
        if all_chunks and self.vectorstore:
            try:
                # Add to vectorstore
                import hashlib
                
                documents = []
                metadatas = []
                ids = []
                
                for i, chunk in enumerate(all_chunks[:20]):  # Limit to 20 chunks
                    doc_id = f"web_{hashlib.md5(f'{drug_name}_{i}'.encode()).hexdigest()}"
                    
                    documents.append(chunk)
                    metadatas.append({
                        "drug_name": drug_name,
                        "source": "web",
                        "chunk_index": i
                    })
                    ids.append(doc_id)
                
                self.vectorstore.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Indexed {len(documents)} web chunks for {drug_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error indexing web content for {drug_name}: {e}")
        
        return False
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        This is the main retrieval function used by the RAG agent.
        
        Args:
            query: User's question
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.vectorstore:
            logger.warning("VectorStore not initialized")
            return []
        
        try:
            results = self.vectorstore.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    retrieved.append({
                        "content": doc,
                        "metadata": metadata,
                        "relevance_score": 1 - distance  # Convert distance to score
                    })
            
            logger.info(f"Retrieved {len(retrieved)} documents for query: {query[:50]}...")
            return retrieved
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def is_drug_known(self, drug_name: str) -> bool:
        """
        Check if a drug exists in our knowledge base.
        
        Args:
            drug_name: Name of the drug to check
            
        Returns:
            True if drug is in knowledge base
        """
        if not self.vectorstore:
            return False
        
        try:
            results = self.vectorstore.query(
                query_texts=[drug_name],
                n_results=1,
                include=["metadatas", "distances"]
            )
            
            if results and results['distances'] and results['distances'][0]:
                # If distance is low (< 0.5), drug is known
                distance = results['distances'][0][0]
                metadata = results['metadatas'][0][0] if results['metadatas'] else {}
                
                # Check if the drug name matches
                known_name = metadata.get('drug_name', '').lower()
                if drug_name.lower() in known_name or known_name in drug_name.lower():
                    return True
                
                # If very close match (distance < 0.3), consider it known
                if distance < 0.3:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if drug known: {e}")
            return False
    
    def fetch_and_cache_drug_sync(self, drug_name: str) -> Optional[str]:
        """
        Synchronously fetch drug info from web and cache it.
        
        Called when a drug is not in the knowledge base.
        Tries drugs.com first, then falls back to RxNorm API.
        
        Args:
            drug_name: Name of the drug to fetch
            
        Returns:
            Fetched content string or None
        """
        import httpx
        from bs4 import BeautifulSoup
        import hashlib
        
        drug_name_lower = drug_name.lower().strip()
        drug_slug = drug_name_lower.replace(' ', '-').replace('_', '-')
        
        # Better headers to bypass blocking
        HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Try multiple URL patterns
        urls_to_try = [
            f"https://www.drugs.com/{drug_slug}.html",
            f"https://www.drugs.com/mtm/{drug_slug}.html",
            f"https://www.drugs.com/cdi/{drug_slug}.html",
            f"https://medlineplus.gov/druginfo/meds/{drug_slug}.html",
        ]
        
        for url in urls_to_try:
            try:
                logger.info(f"Fetching drug info from: {url}")
                
                response = httpx.get(
                    url,
                    timeout=15.0,
                    headers=HEADERS,
                    follow_redirects=True
                )
                
                if response.status_code == 200:
                    # Parse the page
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract useful content
                    content_parts = []
                    
                    # Get title
                    title = soup.find('h1')
                    if title:
                        content_parts.append(f"Drug Name: {title.get_text(strip=True)}")
                    
                    # Get meta description
                    meta_desc = soup.find('meta', {'name': 'description'})
                    if meta_desc:
                        content_parts.append(f"Overview: {meta_desc.get('content', '')}")
                    
                    # Get main content sections
                    for section_name in ['what is', 'uses', 'side effects', 'dosage', 'warnings']:
                        header = soup.find(['h2', 'h3'], string=lambda t: t and section_name in t.lower())
                        if header:
                            next_elem = header.find_next(['p', 'ul'])
                            if next_elem:
                                text = next_elem.get_text(strip=True)[:500]
                                content_parts.append(f"{section_name.title()}: {text}")
                    
                    if content_parts:
                        full_content = "\n\n".join(content_parts)
                        self._cache_drug_content(drug_name, full_content, url)
                        return full_content
                        
            except Exception as e:
                logger.warning(f"Failed to fetch from {url}: {e}")
                continue
        
        # Fallback to RxNorm NIH API (free, no blocking)
        logger.info(f"Trying RxNorm API for: {drug_name}")
        rxnorm_content = self._fetch_from_rxnorm(drug_name)
        if rxnorm_content:
            return rxnorm_content
        
        logger.warning(f"Could not fetch info for: {drug_name}")
        return None
    
    def _fetch_from_rxnorm(self, drug_name: str) -> Optional[str]:
        """Fetch drug info from NIH RxNorm API (always available, free)."""
        import httpx
        import hashlib
        
        try:
            # Search for drug in RxNorm
            search_url = f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={drug_name}"
            logger.info(f"Querying RxNorm API: {search_url}")
            
            response = httpx.get(search_url, timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                
                concepts = data.get('drugGroup', {}).get('conceptGroup', [])
                
                content_parts = [f"Drug Name: {drug_name.title()}"]
                
                for group in concepts:
                    concept_type = group.get('tty', '')
                    concept_props = group.get('conceptProperties', [])
                    
                    if concept_props:
                        for prop in concept_props[:3]:  # Limit to 3
                            name = prop.get('name', '')
                            synonym = prop.get('synonym', '')
                            rxcui = prop.get('rxcui', '')
                            
                            if name:
                                content_parts.append(f"RxNorm Name: {name}")
                            if synonym:
                                content_parts.append(f"Synonym: {synonym}")
                            if rxcui:
                                # Fetch additional info
                                properties = self._fetch_rxnorm_properties(rxcui)
                                if properties:
                                    content_parts.extend(properties)
                
                if len(content_parts) > 1:
                    full_content = "\n".join(content_parts)
                    self._cache_drug_content(drug_name, full_content, "RxNorm API")
                    logger.info(f"[OK] RxNorm: Found info for {drug_name}")
                    return full_content
                    
        except Exception as e:
            logger.warning(f"RxNorm API error: {e}")
        
        return None
    
    def _fetch_rxnorm_properties(self, rxcui: str) -> List[str]:
        """Fetch additional drug properties from RxNorm."""
        import httpx
        
        properties = []
        try:
            # Get drug class
            class_url = f"https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={rxcui}"
            resp = httpx.get(class_url, timeout=5.0)
            if resp.status_code == 200:
                classes = resp.json().get('rxclassDrugInfoList', {}).get('rxclassDrugInfo', [])
                for cls in classes[:2]:
                    class_name = cls.get('rxclassMinConceptItem', {}).get('className', '')
                    if class_name:
                        properties.append(f"Drug Class: {class_name}")
        except:
            pass
        
        return properties
    
    def _cache_drug_content(self, drug_name: str, content: str, source: str):
        """Cache drug content in ChromaDB."""
        import hashlib
        
        if self.vectorstore:
            try:
                doc_id = f"fetched_{hashlib.md5(drug_name.encode()).hexdigest()}"
                
                self.vectorstore.upsert(
                    documents=[content],
                    metadatas=[{
                        "drug_name": drug_name,
                        "source": source,
                    }],
                    ids=[doc_id]
                )
                
                logger.info(f"Cached drug info for: {drug_name} (from {source})")
            except Exception as e:
                logger.warning(f"Failed to cache drug info: {e}")
    
    def get_context_for_question(self, question: str, medicines_to_check: List[str] = None) -> str:
        """
        Get formatted context string for a question.
        
        If medicines are provided and not in knowledge base,
        attempts to fetch from web and cache them.
        
        Args:
            question: User's question
            medicines_to_check: List of medicine names to ensure are in KB
            
        Returns:
            Formatted context string
        """
        # Check if any medicines need to be fetched
        if medicines_to_check:
            for med_name in medicines_to_check[:3]:  # Limit to 3
                med_name_clean = med_name.strip()
                if len(med_name_clean) < 3:
                    continue
                    
                if not self.is_drug_known(med_name_clean):
                    logger.info(f"Drug '{med_name_clean}' not in KB, fetching from web...")
                    fetched = self.fetch_and_cache_drug_sync(med_name_clean)
                    if fetched:
                        logger.info(f"Successfully fetched and cached: {med_name_clean}")
                    else:
                        logger.warning(f"Could not fetch info for: {med_name_clean}")
        
        # Now retrieve with the (possibly updated) knowledge base
        results = self.retrieve(question, n_results=5)
        
        if not results:
            return "No relevant information found in the knowledge base. The LLM will use its general medical knowledge."
        
        context_parts = ["RETRIEVED KNOWLEDGE BASE INFORMATION:\n"]
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            drug_name = metadata.get('drug_name', 'Unknown')
            source = metadata.get('source', 'knowledge_base')
            
            context_parts.append(f"--- Document {i} (Drug: {drug_name}, Source: {source}) ---")
            context_parts.append(result['content'])
            context_parts.append("")
        
        return "\n".join(context_parts)


# Singleton instance
_rag_kb: Optional[LangChainRAGKnowledgeBase] = None


def get_langchain_rag_kb() -> LangChainRAGKnowledgeBase:
    """Get or create the LangChain RAG knowledge base singleton."""
    global _rag_kb
    if _rag_kb is None:
        _rag_kb = LangChainRAGKnowledgeBase()
    return _rag_kb


# LangChain Tool for the agent
def create_rag_retriever_tool():
    """
    Create a LangChain tool for RAG retrieval.
    
    This follows the IBM pattern of creating a retriever tool
    that the agent can use to look up information.
    """
    from langchain.tools import tool
    
    @tool
    def get_drug_information(question: str) -> str:
        """
        Get information about drugs and medications from the medical knowledge base.
        Use this tool when the user asks about drug uses, side effects, dosage, 
        or any medication-related questions.
        """
        kb = get_langchain_rag_kb()
        context = kb.get_context_for_question(question)
        return context
    
    return get_drug_information
