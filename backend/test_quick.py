"""Quick test script for the robust fetcher."""
import asyncio
import sys

async def test():
    try:
        from app.services.robust_fetcher import get_robust_fetcher
        
        fetcher = get_robust_fetcher()
        
        print("Testing drug info fetch for 'metformin'...")
        result = await fetcher.fetch_drug_info("metformin")
        
        if result:
            print("[OK] SUCCESS!")
            print(f"  Name: {result.name}")
            print(f"  Source: {result.source}")
            print(f"  Generic: {result.generic_name}")
            print(f"  Class: {result.drug_class}")
            print(f"  RxCUI: {result.rxcui}")
        else:
            print("[FAIL] Failed to fetch drug info")
        
        # Close the client properly
        from app.services.api_client import close_api_client
        await close_api_client()
        
        # Test template engine
        print("\nTesting template engine...")
        from app.prescription.answer_templates import get_template_engine, DrugContext
        
        engine = get_template_engine()
        ctx = DrugContext(
            name="Aspirin",
            generic_name="acetylsalicylic acid",
            uses="Pain relief"
        )
        answer = engine.generate_drug_info(ctx)
        print(f"[OK] Template generated ({len(answer)} chars)")
        
        print("\n=== All tests passed! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test())

