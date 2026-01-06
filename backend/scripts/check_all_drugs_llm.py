"""
Script to check all drugs in the database using LLM analysis.

This script runs LLM analysis for all drugs in the database for a given patient,
running in parallel with ML predictions.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from app.database import async_session
from app.models import Drug
from app.diabetic.service import DiabeticDDIService, create_diabetic_service
from app.diabetic.llm_drug_checker import get_llm_checker

async def check_all_drugs_for_patient(patient_id: str, limit: int = None):
    """Check all drugs in database for a given patient using LLM."""
    async with async_session() as session:
        service = create_diabetic_service(session)
        
        # Get patient
        patient = await service.get_patient(patient_id)
        if not patient:
            print(f"Patient {patient_id} not found")
            return
        
        # Get current medications
        medications = await service.get_patient_medications(patient_id)
        current_meds = [m.drug_name for m in medications]
        
        # Build patient context
        patient_context = service._build_patient_context(patient)
        
        # Get all drugs from database
        result = await session.execute(select(Drug).limit(limit) if limit else select(Drug))
        drugs = result.scalars().all()
        
        print(f"\n{'='*80}")
        print(f"Checking {len(drugs)} drugs for patient {patient_id}")
        print(f"Patient: {patient.name} (Age: {patient.age}, eGFR: {patient.egfr})")
        print(f"{'='*80}\n")
        
        # Get LLM checker
        llm_checker = get_llm_checker()
        
        # Check if Ollama is available
        available = await llm_checker.is_available()
        if not available:
            print("⚠️  Ollama not available. Install Ollama and pull a model first.")
            print(f"   Recommended: ollama pull {llm_checker.model}")
            return
        
        print(f"✅ Using LLM model: {llm_checker.model}\n")
        
        # Check drugs in batches
        batch_size = 10
        drug_names = [drug.name for drug in drugs[:limit] if limit else drugs]
        
        total = len(drug_names)
        results = []
        
        for i in range(0, total, batch_size):
            batch = drug_names[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} drugs)...")
            
            # Check batch in parallel
            batch_results = await llm_checker.check_multiple_drugs(
                batch, patient_context, current_meds
            )
            
            results.extend(batch_results.items())
            
            # Print summary for this batch
            for drug_name, result in batch_results.items():
                status = "✅" if result.risk_level in ["safe", "caution"] else "⚠️" if result.risk_level == "high_risk" else "🚨"
                print(f"  {status} {drug_name:30s} | Risk: {result.risk_level:15s} | Score: {result.risk_score:5.1f}")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        risk_counts = {}
        for drug_name, result in results:
            risk_counts[result.risk_level] = risk_counts.get(result.risk_level, 0) + 1
        
        for risk_level in ["safe", "caution", "high_risk", "contraindicated", "fatal"]:
            count = risk_counts.get(risk_level, 0)
            if count > 0:
                print(f"  {risk_level:20s}: {count:4d} drugs")
        
        print(f"\nTotal drugs analyzed: {len(results)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python check_all_drugs_llm.py <patient_id> [limit]")
        print("Example: python check_all_drugs_llm.py DEMO001 50")
        sys.exit(1)
    
    patient_id = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    asyncio.run(check_all_drugs_for_patient(patient_id, limit))

