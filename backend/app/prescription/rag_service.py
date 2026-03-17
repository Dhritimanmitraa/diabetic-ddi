"""
RAG Service for Prescription Chat.

Uses ChromaDB for vector storage and retrieval.
"""
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from app.config import get_settings
from app.prescription.schemas import MedicineCreate
from app.services.gemini_client import get_gemini_client

logger = logging.getLogger(__name__)
settings = get_settings()


class PrescriptionRAGService:
    """
    RAG service for prescription question-answering.
    
    Uses ChromaDB for semantic search over prescription content.
    """
    
    def __init__(self):
        """Initialize the RAG service."""
        self.chroma_client = None
        self.embedding_function = None
        self._init_chroma()
    
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
            
            # Use default embedding function (sentence-transformers)
            from chromadb.utils import embedding_functions
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            logger.info(f"ChromaDB initialized at {persist_dir}")
            
        except ImportError as e:
            logger.error(f"ChromaDB not installed: {e}")
        except Exception as e:
            logger.error(f"ChromaDB initialization error: {e}")
    
    def index_prescription(
        self, 
        prescription_id: int, 
        raw_text: str,
        medicines: List[MedicineCreate]
    ) -> Optional[str]:
        """
        Index a prescription in ChromaDB.
        
        Args:
            prescription_id: Database ID of the prescription
            raw_text: Raw extracted text from prescription
            medicines: List of extracted medicines
            
        Returns:
            Collection ID if successful, None otherwise
        """
        if not self.chroma_client:
            logger.warning("ChromaDB not available, skipping indexing")
            return None
        
        collection_name = f"prescription_{prescription_id}"
        
        try:
            # Get or create collection
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"prescription_id": prescription_id}
            )
            
            documents = []
            metadatas = []
            ids = []
            
            # Add raw text as a document
            if raw_text and raw_text.strip():
                documents.append(f"Prescription raw text:\n{raw_text}")
                metadatas.append({"type": "raw_text", "prescription_id": prescription_id})
                ids.append(f"raw_{prescription_id}")
            
            # Add each medicine as a separate document
            for i, med in enumerate(medicines):
                # Create a rich text representation
                med_text = self._medicine_to_text(med)
                documents.append(med_text)
                metadatas.append({
                    "type": "medicine",
                    "medicine_name": med.name,
                    "prescription_id": prescription_id
                })
                ids.append(f"med_{prescription_id}_{i}")
            
            # Add summary document
            if medicines:
                summary = self._create_summary(medicines)
                documents.append(summary)
                metadatas.append({"type": "summary", "prescription_id": prescription_id})
                ids.append(f"summary_{prescription_id}")
            
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Indexed {len(documents)} documents for prescription {prescription_id}")
            
            return collection_name
            
        except Exception as e:
            logger.error(f"Error indexing prescription {prescription_id}: {e}")
            return None
    
    def _medicine_to_text(self, med: MedicineCreate) -> str:
        """Convert medicine to searchable text."""
        parts = [f"Medicine: {med.name}"]
        
        if med.generic_name:
            parts.append(f"Generic name: {med.generic_name}")
        if med.dosage:
            parts.append(f"Dosage: {med.dosage}")
        if med.quantity:
            parts.append(f"Quantity: {med.quantity}")
        if med.frequency:
            parts.append(f"Frequency: {med.frequency}")
        if med.duration:
            parts.append(f"Duration: {med.duration}")
        if med.instructions:
            parts.append(f"Instructions: {med.instructions}")
        
        # Timing
        timing_parts = []
        if med.morning:
            timing_parts.append("morning")
        if med.afternoon:
            timing_parts.append("afternoon")
        if med.evening:
            timing_parts.append("evening")
        if med.night:
            timing_parts.append("night")
        
        if timing_parts:
            parts.append(f"When to take: {', '.join(timing_parts)}")
        
        return "\n".join(parts)
    
    def _create_summary(self, medicines: List[MedicineCreate]) -> str:
        """Create a summary document for all medicines."""
        lines = ["Prescription Summary:", f"Total medicines: {len(medicines)}", ""]
        
        for i, med in enumerate(medicines, 1):
            timing = []
            if med.morning:
                timing.append("morning")
            if med.afternoon:
                timing.append("afternoon")
            if med.evening:
                timing.append("evening")
            if med.night:
                timing.append("night")
            
            freq_str = f" ({', '.join(timing)})" if timing else ""
            dose_str = f" {med.dosage}" if med.dosage else ""
            
            lines.append(f"{i}. {med.name}{dose_str}{freq_str}")
        
        return "\n".join(lines)
    
    def query(
        self, 
        prescription_id: int, 
        question: str,
        n_results: int = 3
    ) -> str:
        """
        Query the prescription for relevant context.
        
        Args:
            prescription_id: Database ID of the prescription
            question: User's question
            n_results: Number of results to retrieve
            
        Returns:
            Retrieved context string
        """
        if not self.chroma_client:
            return ""
        
        collection_name = f"prescription_{prescription_id}"
        
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            results = collection.query(
                query_texts=[question],
                n_results=n_results
            )
            
            if results and results['documents']:
                # Combine retrieved documents
                context = "\n\n---\n\n".join(results['documents'][0])
                return context
            
            return ""
            
        except Exception as e:
            logger.error(f"Error querying prescription {prescription_id}: {e}")
            return ""
    
    def delete_prescription(self, prescription_id: int) -> bool:
        """Delete a prescription's index from ChromaDB."""
        if not self.chroma_client:
            return False
        
        collection_name = f"prescription_{prescription_id}"
        
        try:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted ChromaDB collection {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"Error deleting collection {collection_name}: {e}")
            return False


class PrescriptionLLMService:
    """
    LLM service for answering questions about prescriptions.
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self.gemini_client = None
        self.gemini_available = False
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Google Gemini client."""
        self.gemini_client = get_gemini_client("gemini-2.0-flash")
        self.gemini_available = self.gemini_client.is_available
        if self.gemini_available:
            logger.info(f"Gemini LLM initialized for chat via {self.gemini_client.sdk}")
    
    async def answer_question(
        self, 
        question: str, 
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        use_llm: bool = True
    ) -> tuple[str, str]:
        """
        Answer a question about a prescription.
        
        Args:
            question: User's question
            context: Retrieved context from RAG
            chat_history: Previous chat messages
            use_llm: Whether to use LLM (if False, uses templates)
            
        Returns:
            Tuple of (answer, model_used)
        """
        system_prompt = """You are a prescription-support assistant.

IMPORTANT RULES:
1. Answer only about the uploaded prescription, extracted medicines, and high-level medication safety information.
2. If the needed fact is not present in the prescription context, say that clearly instead of guessing.
3. Do not diagnose, prescribe, choose treatment plans, or tell the user to start, stop, increase, or decrease a medicine.
4. Refuse questions outside prescription and medication-information scope.
5. For urgent symptoms, overdose concerns, allergic reactions, trouble breathing, chest pain, severe bleeding, seizures, or loss of consciousness, tell the user to seek emergency medical care immediately.
6. If asked about interactions, side effects, pregnancy, kidney or liver disease, or other higher-risk concerns, provide only cautious general information and advise confirming with a clinician or pharmacist.
7. Be concise, factual, and non-alarmist.

PRESCRIPTION CONTEXT:
{context}

Answer the user's question using only the context above. If the answer is outside scope or not supported by the context, say so clearly."""

        full_prompt = system_prompt.format(context=context if context else "No prescription context available.")
        
        # If LLM is disabled or fallback mode, use templates
        if not use_llm or (settings.LLM_FALLBACK_TO_TEMPLATES and not self.gemini_available):
            return self._answer_with_templates(question, context)
        
        # Try Gemini first
        if self.gemini_available:
            try:
                return await self._answer_with_gemini(question, full_prompt, chat_history)
            except Exception as e:
                logger.warning(f"Gemini chat error: {e}")
        
        # Try Ollama
        try:
            return await self._answer_with_ollama(question, full_prompt, chat_history)
        except Exception as e:
            logger.warning(f"Ollama chat error: {e}")
            
            # Final fallback to templates
            if settings.LLM_FALLBACK_TO_TEMPLATES:
                logger.info("Using template-based answer as LLM fallback")
                return self._answer_with_templates(question, context)
            
            return "I'm sorry, I couldn't process your question. Please try again.", "error"
    
    def _answer_with_templates(
        self,
        question: str,
        context: str
    ) -> tuple[str, str]:
        """Answer using template engine when LLM is unavailable."""
        try:
            from app.prescription.answer_templates import get_template_engine
            
            engine = get_template_engine()
            
            # Parse context into structured format for templates
            rag_documents = []
            if context:
                # Simple parsing of context
                parts = context.split("---")
                for part in parts:
                    if part.strip():
                        rag_documents.append({
                            "content": part.strip(),
                            "metadata": {"source": "prescription"}
                        })
            
            answer = engine.generate_from_rag_context(question, rag_documents)
            return answer, "template_engine"
            
        except Exception as e:
            logger.error(f"Template engine error: {e}")
            return self._generate_basic_answer(question, context), "basic_fallback"
    
    def _generate_basic_answer(self, question: str, context: str) -> str:
        """Generate a very basic answer when everything else fails."""
        if not context:
            return f"""## Your Question: {question}

I couldn't find specific information about this in your prescription.

Please consult your healthcare provider for accurate information about your medications."""
        
        return f"""## Your Question: {question}

Based on your prescription, here's the relevant information:

{context[:1000]}

---
*For specific medical advice, please consult your healthcare provider.*"""
    
    async def _answer_with_gemini(
        self, 
        question: str, 
        system_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[str, str]:
        """Answer using Gemini."""
        # Build conversation
        messages = [system_prompt]
        
        if chat_history:
            for msg in chat_history[-6:]:  # Last 6 messages for context
                messages.append(f"{msg['role'].upper()}: {msg['content']}")
        
        messages.append(f"USER: {question}")
        
        response = self.gemini_client.generate_text(
            "\n\n".join(messages),
            temperature=0.3,
            max_output_tokens=1024,
        )
        
        return response.text, response.model
    
    async def _answer_with_ollama(
        self, 
        question: str, 
        system_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[str, str]:
        """Answer using Ollama."""
        import ollama
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg['role'], "content": msg['content']})
        
        messages.append({"role": "user", "content": question})
        
        response = ollama.chat(
            model="gpt-oss:120b-cloud",
            messages=messages,
            options={
                'temperature': 0.3,
                'num_predict': 1024,
            }
        )
        
        return response['message']['content'], "gpt-oss:120b-cloud"


# Singleton instances
_rag_service: Optional[PrescriptionRAGService] = None
_llm_service: Optional[PrescriptionLLMService] = None


def get_rag_service() -> PrescriptionRAGService:
    """Get or create the RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = PrescriptionRAGService()
    return _rag_service


def get_llm_service() -> PrescriptionLLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = PrescriptionLLMService()
    return _llm_service
