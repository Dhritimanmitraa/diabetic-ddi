"""
Drug Knowledge Base Service.

Indexes drug information in ChromaDB for RAG retrieval.
"""
import logging
import os
import json
from typing import Optional, List, Dict, Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DrugKnowledgeBase:
    """
    Knowledge base service for drug information retrieval.
    
    Indexes drug knowledge in ChromaDB and provides semantic search.
    """
    
    COLLECTION_NAME = "drug_knowledge"
    KNOWLEDGE_FILE = "data/drug_knowledge.json"
    
    def __init__(self):
        """Initialize the knowledge base."""
        self.chroma_client = None
        self.embedding_function = None
        self.collection = None
        self.drugs_data = []
        self._init_chroma()
        self._load_and_index_knowledge()
    
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
            
            logger.info("Drug Knowledge Base ChromaDB initialized")
            
        except Exception as e:
            logger.error(f"ChromaDB initialization error: {e}")
    
    def _load_and_index_knowledge(self):
        """Load drug knowledge from JSON and index in ChromaDB."""
        if not self.chroma_client:
            logger.warning("ChromaDB not available, skipping knowledge indexing")
            return
        
        # Load drug knowledge JSON
        knowledge_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            self.KNOWLEDGE_FILE
        )
        
        if not os.path.exists(knowledge_path):
            logger.warning(f"Drug knowledge file not found: {knowledge_path}")
            return
        
        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.drugs_data = data.get('drugs', [])
            
            logger.info(f"Loaded {len(self.drugs_data)} drugs from knowledge base")
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"type": "drug_knowledge"}
            )
            
            # Check if already indexed
            existing = self.collection.count()
            if existing >= len(self.drugs_data):
                logger.info(f"Drug knowledge already indexed ({existing} documents)")
                return
            
            # Index each drug
            documents = []
            metadatas = []
            ids = []
            
            for i, drug in enumerate(self.drugs_data):
                # Create rich searchable text
                doc_text = f"""
Drug Name: {drug.get('name', '')}
Generic Name: {drug.get('generic_name', '')}
Drug Class: {drug.get('drug_class', '')}
Uses: {drug.get('uses', '')}
How It Works: {drug.get('mechanism', '')}
Side Effects: {drug.get('side_effects', '')}
Dosage: {drug.get('dosage', '')}
Precautions: {drug.get('precautions', '')}
""".strip()
                
                documents.append(doc_text)
                metadatas.append({
                    "name": drug.get('name', ''),
                    "generic_name": drug.get('generic_name', ''),
                    "drug_class": drug.get('drug_class', ''),
                    "type": "drug_info"
                })
                ids.append(f"drug_{i}_{drug.get('name', 'unknown').lower().replace(' ', '_')}")
            
            # Add to collection
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Indexed {len(documents)} drugs in knowledge base")
            
        except Exception as e:
            logger.error(f"Error loading drug knowledge: {e}")
    
    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search the drug knowledge base.
        
        Args:
            query: Search query (e.g., "What does Biopan do?")
            n_results: Number of results to return
            
        Returns:
            List of matching drug documents with metadata
        """
        if not self.collection:
            logger.warning("Knowledge base not initialized")
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
            
            logger.info(f"Knowledge search for '{query[:50]}...': {len(matches)} results")
            return matches
            
        except Exception as e:
            logger.error(f"Knowledge search error: {e}")
            return []
    
    def get_drug_by_name(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Get drug information by exact or partial name match.
        
        Args:
            drug_name: Name of the drug to look up
            
        Returns:
            Drug information dict if found, None otherwise
        """
        drug_name_lower = drug_name.lower()
        
        for drug in self.drugs_data:
            if (drug_name_lower in drug.get('name', '').lower() or
                drug_name_lower in drug.get('generic_name', '').lower()):
                return drug
        
        return None


# Singleton instance
_knowledge_base: Optional[DrugKnowledgeBase] = None


def get_drug_knowledge_base() -> DrugKnowledgeBase:
    """Get or create the drug knowledge base singleton."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = DrugKnowledgeBase()
    return _knowledge_base
