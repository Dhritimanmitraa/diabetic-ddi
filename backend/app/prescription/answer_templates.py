"""
Answer Templates for LLM-free response generation.

This module provides template-based answer generation that works
WITHOUT any LLM dependency. Falls back to these templates when
Ollama or other LLM services are unavailable.
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DrugContext:
    """Context for drug-related templates."""
    name: str
    generic_name: Optional[str] = None
    drug_class: Optional[str] = None
    uses: Optional[str] = None
    mechanism: Optional[str] = None
    side_effects: Optional[str] = None
    warnings: Optional[str] = None
    dosage: Optional[str] = None
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DrugContext":
        return cls(
            name=data.get("name", "Unknown Drug"),
            generic_name=data.get("generic_name"),
            drug_class=data.get("drug_class"),
            uses=data.get("uses"),
            mechanism=data.get("mechanism"),
            side_effects=data.get("side_effects"),
            warnings=data.get("warnings"),
            dosage=data.get("dosage"),
            description=data.get("description"),
        )


@dataclass
class InteractionContext:
    """Context for interaction-related templates."""
    drug1: str
    drug2: str
    severity: str
    effect: Optional[str] = None
    mechanism: Optional[str] = None
    recommendation: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionContext":
        return cls(
            drug1=data.get("drug1", "Drug 1"),
            drug2=data.get("drug2", "Drug 2"),
            severity=data.get("severity", "unknown"),
            effect=data.get("effect"),
            mechanism=data.get("mechanism"),
            recommendation=data.get("recommendation"),
        )


# ============ Answer Templates ============

DRUG_INFO_TEMPLATE = """## {name}
{generic_line}
{class_line}

### Overview
{description_or_default}

### Uses & Indications
{uses_or_default}

### How It Works
{mechanism_or_default}

### Common Side Effects
{side_effects_or_default}

### Important Warnings
{warnings_or_default}

### Dosage Information
{dosage_or_default}

---
*This information is for educational purposes only. Always consult your healthcare provider before making medication decisions.*
"""

INTERACTION_ALERT_TEMPLATE = """## Drug Interaction Alert

**{drug1}** + **{drug2}** = **{severity_display}** Risk

### Effect
{effect_or_default}

### Clinical Significance
{significance}

### Recommendation
{recommendation_or_default}

### Action Required
{action}

---
*Always consult your healthcare provider about drug interactions.*
"""

SAFE_COMBINATION_TEMPLATE = """## Drug Combination Check

**{drug1}** + **{drug2}**

### Result
No significant interaction found between these medications.

### Note
While no major interaction is documented, you should:
- Monitor for any unusual symptoms
- Inform your healthcare provider of all medications
- Follow prescribed dosing schedules

### General Precautions
- Take medications as directed
- Report any unexpected side effects
- Keep an updated list of all your medications

---
*This does not guarantee safety. Consult your healthcare provider.*
"""

UNKNOWN_DRUG_TEMPLATE = """## Drug Information Not Found

Sorry, I couldn't find detailed information for **{name}**.

### Possible Reasons
- The drug name may be misspelled
- It might be a brand name not in our database
- It could be a very new or specialized medication

### Suggestions
1. Check the spelling of the drug name
2. Try the generic name instead of brand name
3. Consult your pharmacist or healthcare provider

### General Safety Advice
Always verify medication information with your healthcare provider or pharmacist before use.
"""

DIABETIC_RISK_TEMPLATE = """## Diabetic Patient Drug Assessment

### Drug: {drug_name}
### Risk Level: {risk_level_display}

### Patient Factors Considered
{patient_factors}

### Risk Factors
{risk_factors}

### Kidney Function Impact
{kidney_impact}

### Recommended Actions
{recommendations}

### Monitoring Required
{monitoring}

### Safer Alternatives
{alternatives}

---
*Diabetic patients require special medication considerations. Always consult your endocrinologist.*
"""


class AnswerTemplateEngine:
    """
    Template-based answer engine for LLM-free response generation.
    
    Usage:
        engine = AnswerTemplateEngine()
        response = engine.generate_drug_info(drug_context)
    """
    
    SEVERITY_DISPLAY = {
        "contraindicated": "CONTRAINDICATED",
        "major": "MAJOR",
        "moderate": "MODERATE",
        "minor": "MINOR",
        "none": "NONE",
        "unknown": "UNKNOWN",
    }
    
    SEVERITY_SIGNIFICANCE = {
        "contraindicated": "These medications should NEVER be used together. The combination can cause life-threatening effects.",
        "major": "This interaction can cause serious harm. Alternative medications should be considered.",
        "moderate": "This interaction may worsen existing conditions or reduce drug effectiveness. Medical supervision recommended.",
        "minor": "This interaction is unlikely to cause significant problems but should be monitored.",
        "none": "No significant interaction expected.",
        "unknown": "Interaction potential is unclear. Exercise caution.",
    }
    
    SEVERITY_ACTION = {
        "contraindicated": "DO NOT USE TOGETHER. Seek immediate medical consultation for alternatives.",
        "major": "Consult your doctor before combining. Alternatives may be needed.",
        "moderate": "Use with caution. Monitor for side effects and report any concerns.",
        "minor": "Generally safe but stay alert for any unusual symptoms.",
        "none": "No special precautions needed for this combination.",
        "unknown": "Consult your pharmacist or doctor for guidance.",
    }
    
    RISK_LEVEL_DISPLAY = {
        "safe": "SAFE",
        "caution": "USE WITH CAUTION",
        "high_risk": "HIGH RISK",
        "contraindicated": "CONTRAINDICATED",
        "fatal": "POTENTIALLY FATAL",
    }
    
    def generate_drug_info(self, context: DrugContext) -> str:
        """Generate drug information response."""
        return DRUG_INFO_TEMPLATE.format(
            name=context.name,
            generic_line=f"**Generic Name:** {context.generic_name}" if context.generic_name else "",
            class_line=f"**Drug Class:** {context.drug_class}" if context.drug_class else "",
            description_or_default=context.description or f"{context.name} is a medication used in clinical practice.",
            uses_or_default=context.uses or "Consult your healthcare provider for specific indications.",
            mechanism_or_default=context.mechanism or "Works through various pharmacological mechanisms. Ask your pharmacist for details.",
            side_effects_or_default=context.side_effects or "Side effects vary by individual. Report any unusual symptoms to your doctor.",
            warnings_or_default=context.warnings or "Follow your doctor's instructions. Inform them of all other medications you take.",
            dosage_or_default=context.dosage or "Follow the dosage prescribed by your healthcare provider.",
        ).strip()
    
    def generate_interaction_alert(self, context: InteractionContext) -> str:
        """Generate interaction alert response."""
        severity = (context.severity or "unknown").lower()
        
        return INTERACTION_ALERT_TEMPLATE.format(
            drug1=context.drug1,
            drug2=context.drug2,
            severity_display=self.SEVERITY_DISPLAY.get(severity, self.SEVERITY_DISPLAY["unknown"]),
            effect_or_default=context.effect or "May alter the effects of one or both medications.",
            significance=self.SEVERITY_SIGNIFICANCE.get(severity, self.SEVERITY_SIGNIFICANCE["unknown"]),
            recommendation_or_default=context.recommendation or "Discuss this combination with your healthcare provider.",
            action=self.SEVERITY_ACTION.get(severity, self.SEVERITY_ACTION["unknown"]),
        ).strip()
    
    def generate_safe_combination(self, drug1: str, drug2: str) -> str:
        """Generate safe combination response."""
        return SAFE_COMBINATION_TEMPLATE.format(
            drug1=drug1,
            drug2=drug2,
        ).strip()
    
    def generate_unknown_drug(self, drug_name: str) -> str:
        """Generate unknown drug response."""
        return UNKNOWN_DRUG_TEMPLATE.format(name=drug_name).strip()
    
    def generate_diabetic_risk_assessment(
        self,
        drug_name: str,
        risk_level: str,
        patient_factors: List[str],
        risk_factors: List[str],
        kidney_impact: str,
        recommendations: List[str],
        monitoring: List[str],
        alternatives: List[str],
    ) -> str:
        """Generate diabetic patient risk assessment."""
        return DIABETIC_RISK_TEMPLATE.format(
            drug_name=drug_name,
            risk_level_display=self.RISK_LEVEL_DISPLAY.get(risk_level, risk_level.upper()),
            patient_factors="- " + "\n- ".join(patient_factors) if patient_factors else "- No specific factors identified",
            risk_factors="- " + "\n- ".join(risk_factors) if risk_factors else "- No significant risks identified",
            kidney_impact=kidney_impact or "Monitor kidney function as appropriate.",
            recommendations="- " + "\n- ".join(recommendations) if recommendations else "- Follow standard prescribing guidelines",
            monitoring="- " + "\n- ".join(monitoring) if monitoring else "- Routine monitoring",
            alternatives="- " + "\n- ".join(alternatives) if alternatives else "- Consult your doctor for alternatives if needed",
        ).strip()
    
    def generate_from_rag_context(
        self,
        question: str,
        rag_documents: List[Dict[str, Any]],
    ) -> str:
        """
        Generate answer from RAG context without LLM.
        
        Extracts relevant information from retrieved documents
        and formats it into a coherent response.
        """
        if not rag_documents:
            return self._generate_no_context_response(question)
        
        # Combine relevant content
        combined_content = []
        sources = set()
        
        for doc in rag_documents[:5]:  # Top 5 documents
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "knowledge_base")
            # drug_name available in metadata if needed for future enhancements
            
            if content:
                combined_content.append(content)
            if source:
                sources.add(source)
        
        if not combined_content:
            return self._generate_no_context_response(question)
        
        # Format response
        response_parts = [
            f"## Answer to: {question}\n",
            "Based on our medical knowledge base:\n",
        ]
        
        # Add relevant content
        for i, content in enumerate(combined_content[:3], 1):
            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."
            response_parts.append(f"### Finding {i}\n{content}\n")
        
        # Add sources
        if sources:
            response_parts.append(f"\n**Sources:** {', '.join(sources)}")
        
        response_parts.append("\n\n---\n*Always verify medical information with healthcare professionals.*")
        
        return "\n".join(response_parts)
    
    def _generate_no_context_response(self, question: str) -> str:
        """Generate response when no context is available."""
        return f"""## {question}

I don't have specific information to answer this question in my knowledge base.

### Suggestions
1. Try rephrasing your question
2. Ask about a specific drug by name
3. Consult your healthcare provider for personalized advice

### General Health Reminder
- Always consult qualified healthcare professionals for medical decisions
- Never stop or change medications without medical guidance
- Keep your healthcare providers informed of all medications you take

---
*This is an automated response. For accurate medical information, please consult a healthcare professional.*
"""


# Singleton instance
_engine: Optional[AnswerTemplateEngine] = None


def get_template_engine() -> AnswerTemplateEngine:
    """Get or create the template engine singleton."""
    global _engine
    if _engine is None:
        _engine = AnswerTemplateEngine()
    return _engine
