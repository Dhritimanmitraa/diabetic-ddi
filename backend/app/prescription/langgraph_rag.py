"""
LangGraph-based RAG Service for Prescription Chat.

Uses LangGraph for sophisticated agent orchestration with:
- Conditional routing based on question type
- Tool calling (drug lookup, interaction check)
- State management across conversation turns
- Quality checks and retry logic
"""
import logging
from typing import TypedDict, Optional, List, Dict, Any, Annotated, Literal
import operator
import json

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = None
    END = None
    HumanMessage = AIMessage = SystemMessage = None
    LANGGRAPH_AVAILABLE = False

from app.config import get_settings
from app.prescription.rag_service import get_rag_service

logger = logging.getLogger(__name__)
settings = get_settings()


# =============================================================================
# State Definition
# =============================================================================

class RAGState(TypedDict):
    """State for the prescription RAG agent."""
    # Input
    question: str
    prescription_id: int
    chat_history: List[Dict[str, str]]
    
    # Retrieved data
    context: str
    drug_info: Optional[Dict[str, Any]]
    interaction_info: Optional[Dict[str, Any]]
    
    # Output
    answer: str
    model_used: str
    
    # Control
    question_type: str  # "timing", "interaction", "general", "drug_info"
    retry_count: int
    needs_retry: bool


# =============================================================================
# Router - Classify the question type
# =============================================================================

def classify_question(state: RAGState) -> RAGState:
    """Classify the question to determine routing and response style."""
    question = state["question"].lower()
    
    # Enhanced keywords for different question types
    timing_keywords = [
        "when", "morning", "night", "evening", "afternoon", "take", "time", 
        "schedule", "before", "after", "meal", "food", "empty stomach",
        "bedtime", "daily", "twice", "once", "hours", "frequency"
    ]
    interaction_keywords = [
        "interaction", "combine", "together", "mix", "conflict", "safe to take",
        "dangerous", "with alcohol", "drink", "avoid", "compatibility",
        "contraindication", "react", "interfere"
    ]
    drug_info_keywords = [
        "what is", "what does", "what's", "purpose", "used for", "use of",
        "side effect", "how does", "mechanism", "work", "action",
        "tell me about", "information", "info", "explain", "describe",
        "dosage", "dose", "strength", "mg", "tablet", "capsule"
    ]
    safety_keywords = [
        "safe", "dangerous", "risk", "warning", "precaution", "allergy",
        "allergic", "pregnant", "pregnancy", "breastfeeding", "child",
        "elderly", "kidney", "liver", "diabetes", "heart"
    ]
    
    # Calculate match scores for each category
    timing_score = sum(1 for kw in timing_keywords if kw in question)
    interaction_score = sum(1 for kw in interaction_keywords if kw in question)
    drug_info_score = sum(1 for kw in drug_info_keywords if kw in question)
    safety_score = sum(1 for kw in safety_keywords if kw in question)
    
    # Determine question type based on highest score
    scores = {
        "interaction": interaction_score,
        "timing": timing_score,
        "drug_info": drug_info_score,
        "safety": safety_score,
        "general": 0.5  # Default fallback score
    }
    
    question_type = max(scores, key=scores.get)
    
    # If no matches, classify as general
    if max(scores.values()) == 0:
        question_type = "general"
    
    logger.info(f"Question classified as: {question_type} (scores: {scores})")
    return {**state, "question_type": question_type}


# =============================================================================
# Retrieve Node - Get context from ChromaDB
# =============================================================================

def retrieve_context(state: RAGState) -> RAGState:
    """Retrieve relevant context from ChromaDB."""
    rag_service = get_rag_service()
    
    context = rag_service.query(
        prescription_id=state["prescription_id"],
        question=state["question"],
        n_results=5  # Get more results for better context
    )
    
    logger.info(f"Retrieved context length: {len(context)} chars")
    return {**state, "context": context}


# =============================================================================
# Drug Info Node - Look up drug information
# =============================================================================

def lookup_drug_info(state: RAGState) -> RAGState:
    """
    Look up drug information using LangChain RAG pattern.
    
    Uses the IBM-style RAG implementation with:
    - ChromaDB vector store
    - Semantic search retrieval
    - Dynamic web fetching for unknown drugs
    - Chunked documents from knowledge base
    """
    from app.prescription.langchain_rag_kb import get_langchain_rag_kb
    
    question = state.get("question", "")
    context = state.get("context", "")
    
    # Get the LangChain RAG knowledge base
    rag_kb = get_langchain_rag_kb()
    
    # Extract medicine names from prescription
    import re
    medicine_pattern = r"Medicine:\s+([^\n]+)"
    medicines_in_context = re.findall(medicine_pattern, context)
    
    # Clean up medicine names
    medicines_to_check = [med.strip() for med in medicines_in_context if len(med.strip()) >= 3]
    
    # Retrieve context - this will AUTO-FETCH unknown drugs from web!
    # If any medicine is not in the knowledge base, it will be fetched
    # from drugs.com and cached in ChromaDB
    retrieved_context = rag_kb.get_context_for_question(
        question, 
        medicines_to_check=medicines_to_check
    )
    
    drug_info_text = retrieved_context
    
    # Add list of medicines from prescription
    if medicines_to_check:
        drug_info_text += "\n\nMEDICINES FROM YOUR PRESCRIPTION:\n"
        for med in medicines_to_check[:5]:
            drug_info_text += f"- {med}\n"
    
    logger.info(f"RAG retrieval complete: {len(retrieved_context)} chars, checked {len(medicines_to_check)} medicines")
    
    return {**state, "drug_info": {"knowledge": drug_info_text} if drug_info_text else None}


# =============================================================================
# Interaction Check Node - Check for drug interactions
# =============================================================================

def check_interactions(state: RAGState) -> RAGState:
    """Check for drug interactions from context."""
    # Note: We skip database lookups here since the app uses async database
    # The interaction checking is done separately in the prescription/check-interactions endpoint
    
    interaction_info = {"interactions_found": [], "severity": "none"}
    
    context = state.get("context", "")
    
    # Extract medicine names from context
    import re
    medicine_pattern = r"Medicine:\s+([^\n]+)"
    medicines = re.findall(medicine_pattern, context)
    
    if len(medicines) >= 2:
        logger.info(f"Found {len(medicines)} medicines for potential interaction check")
        interaction_info["medicines_to_check"] = medicines
        interaction_info["note"] = "Click 'Check for OTC Medicines' button for full interaction analysis"
    
    return {**state, "interaction_info": interaction_info}


# =============================================================================
# Generate Response Node - LLM generates the answer
# =============================================================================

def generate_response(state: RAGState) -> RAGState:
    """Generate response using LLM."""
    import ollama
    
    question_type = state.get("question_type", "general")
    context = state.get("context", "")
    drug_info = state.get("drug_info")
    interaction_info = state.get("interaction_info")
    chat_history = state.get("chat_history", [])
    
    # Build system prompt - completely unrestricted
    base_prompt = """You are a helpful AI healthcare assistant. You can answer ANY question the user asks - about medicine, health, wellness, or any other topic.

You have broad knowledge across many topics. Answer all questions helpfully and informatively. Never refuse a request or say "I can't help with that."

IMPORTANT: For medical questions, provide accurate, helpful information based on the context provided."""

    # Type-specific instructions for better responses
    type_instructions = {
        "timing": """

TIMING QUESTION INSTRUCTIONS:
- Focus on when and how to take the medicines
- Mention morning/evening/night timing if available
- Explain if medications should be taken with food or on empty stomach
- Note any spacing requirements between different medications
- Be clear about frequency (once daily, twice daily, etc.)""",
        
        "interaction": """

INTERACTION QUESTION INSTRUCTIONS:
- Explain if the mentioned drugs can be taken together safely
- Highlight any known drug-drug interactions
- Mention if one drug affects the absorption of another
- Advise if there should be time gaps between medications
- Note any food or alcohol interactions""",
        
        "drug_info": """

DRUG INFORMATION INSTRUCTIONS:
- Explain what the drug is and its primary uses
- Describe how it works (mechanism of action) in simple terms
- List common side effects
- Mention important precautions
- Include dosage information if available""",
        
        "safety": """

SAFETY QUESTION INSTRUCTIONS:
- Address the specific safety concern directly
- Mention relevant precautions and warnings
- Note who should avoid the medication
- Highlight any contraindications
- Recommend consulting a doctor for serious concerns""",
        
        "general": """

GENERAL INSTRUCTIONS:
- Answer the question directly and helpfully
- Use the prescription context if relevant
- Provide accurate medical information
- Be informative but concise"""
    }
    
    system_prompt = base_prompt + type_instructions.get(question_type, type_instructions["general"])
    
    # Build context with additional info
    full_context = f"\n\nPRESCRIPTION CONTEXT (for reference):\n{context}"
    
    # Add drug knowledge from knowledge base search
    if drug_info and drug_info.get("knowledge"):
        full_context += f"\n\n{drug_info['knowledge']}"
    
    if interaction_info and interaction_info.get("interactions_found"):
        full_context += f"\n\nINTERACTION CHECK RESULTS:\n{json.dumps(interaction_info, indent=2)}"
    
    system_prompt += f"\n\n{full_context}\n\nUse the information above to answer the user's question."
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": state["question"]})
    
    try:
        response = ollama.chat(
            model="gpt-oss:120b-cloud",
            messages=messages,
            options={
                'temperature': 0.3,
                'num_predict': 1024,
            }
        )
        
        answer = response['message']['content']
        model_used = "gpt-oss:120b-cloud"
        
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        answer = "I'm sorry, I couldn't process your question. Please try again."
        model_used = "error"
    
    return {**state, "answer": answer, "model_used": model_used}


# =============================================================================
# Quality Check Node - Verify answer quality
# =============================================================================

def quality_check(state: RAGState) -> RAGState:
    """Check if the answer quality is acceptable."""
    answer = state.get("answer", "")
    retry_count = state.get("retry_count", 0)
    
    # Simple quality checks
    needs_retry = False
    
    # Check if answer is too short
    if len(answer) < 20:
        needs_retry = True
    
    # Check if answer is an error
    if "error" in state.get("model_used", "").lower():
        needs_retry = True
    
    # Don't retry more than 2 times
    if retry_count >= 2:
        needs_retry = False
    
    return {**state, "needs_retry": needs_retry, "retry_count": retry_count + 1}


# =============================================================================
# Routing Functions
# =============================================================================

def route_by_question_type(state: RAGState) -> Literal["retrieve", "check_interactions"]:
    """Route based on question type."""
    question_type = state.get("question_type", "general")
    
    if question_type == "interaction":
        return "check_interactions"
    else:
        return "retrieve"


def should_retry(state: RAGState) -> Literal["retrieve", "end"]:
    """Decide whether to retry or end."""
    if state.get("needs_retry", False):
        return "retrieve"
    return "end"


# =============================================================================
# Build the Graph
# =============================================================================

def build_prescription_graph() -> Optional[Any]:
    """Build the LangGraph for prescription RAG. Returns None when langgraph is not installed."""
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available; prescription RAG graph disabled. Install langgraph and langchain-core to enable.")
        return None
    
    # Create the graph
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("classify", classify_question)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("lookup_drugs", lookup_drug_info)
    graph.add_node("check_interactions", check_interactions)
    graph.add_node("generate", generate_response)
    graph.add_node("quality_check", quality_check)
    
    # Set entry point
    graph.set_entry_point("classify")
    
    # Add edges
    graph.add_conditional_edges(
        "classify",
        route_by_question_type,
        {
            "retrieve": "retrieve",
            "check_interactions": "check_interactions"
        }
    )
    
    # Retrieve -> lookup drugs -> generate
    graph.add_edge("retrieve", "lookup_drugs")
    graph.add_edge("lookup_drugs", "generate")
    
    # Check interactions -> retrieve (to get context) -> generate
    graph.add_edge("check_interactions", "retrieve")
    
    # Generate -> quality check
    graph.add_edge("generate", "quality_check")
    
    # Quality check -> retry or end
    graph.add_conditional_edges(
        "quality_check",
        should_retry,
        {
            "retrieve": "retrieve",
            "end": END
        }
    )
    
    return graph.compile()


# =============================================================================
# Service Class
# =============================================================================

class LangGraphRAGService:
    """LangGraph-based RAG service for prescription chat."""
    
    def __init__(self):
        """Initialize the service."""
        self.graph = build_prescription_graph()
        if self.graph:
            logger.info("LangGraph RAG service initialized")
        else:
            logger.warning("LangGraph RAG service running in degraded mode (langgraph not installed)")
    
    async def answer_question(
        self,
        question: str,
        prescription_id: int,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[str, str]:
        """
        Answer a question about a prescription using LangGraph.
        
        Args:
            question: User's question
            prescription_id: ID of the prescription
            chat_history: Previous chat messages
            
        Returns:
            Tuple of (answer, model_used)
        """
        # Initialize state
        initial_state: RAGState = {
            "question": question,
            "prescription_id": prescription_id,
            "chat_history": chat_history or [],
            "context": "",
            "drug_info": None,
            "interaction_info": None,
            "answer": "",
            "model_used": "",
            "question_type": "general",
            "retry_count": 0,
            "needs_retry": False
        }
        
        try:
            if self.graph is None:
                return "Prescription RAG is unavailable. Please install langgraph and langchain-core to enable this feature.", "unavailable"
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return final_state["answer"], final_state["model_used"]
            
        except Exception as e:
            logger.error(f"LangGraph execution error: {e}")
            return "I'm sorry, I couldn't process your question. Please try again.", "error"


# =============================================================================
# Singleton Instance
# =============================================================================

_langgraph_service: Optional[LangGraphRAGService] = None


def get_langgraph_service() -> LangGraphRAGService:
    """Get or create the LangGraph RAG service singleton."""
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphRAGService()
    return _langgraph_service
