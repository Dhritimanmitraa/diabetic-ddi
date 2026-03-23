import asyncio
import sys
import os

# Add backend to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app.main  # Bootstrap all routes and models
from app.database import async_session
from app.services.interaction_service import InteractionService

async def test_api():
    print("Starting API test...")
    async with async_session() as session:
        service = InteractionService(session)
        # Test 1: Single interaction
        print("\n--- Testing Single Interaction (Warfarin + Aspirin) ---")
        response = await service.check_interaction('Warfarin', 'Aspirin')
        print(f"Result: {response.has_interaction}")
        if response.has_interaction and response.interaction:
            print(f"Severity: {response.interaction.severity}")
            print(f"Description: {response.interaction.description}")
            print(f"Management: {response.interaction.management}")
        
        # Test 2: Batch interactions
        print("\n--- Testing Batch Interactions ---")
        drugs = ['Warfarin', 'Aspirin', 'Metoprolol']
        print(f"Checking list: {drugs}")
        batch_responses = await service.check_batch_interactions(drugs)
        for (d1, d2, resp) in batch_responses:
            if resp.has_interaction and resp.interaction:
                print(f"[!] {d1} + {d2} -> {resp.interaction.severity.value}")
            else:
                print(f"[ ] {d1} + {d2} -> Safe/No Interaction")

if __name__ == "__main__":
    asyncio.run(test_api())
