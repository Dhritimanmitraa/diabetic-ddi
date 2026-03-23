import asyncio
import sys
import os
import gc
import pandas as pd
from sqlalchemy import text, insert, select, func
from sqlalchemy.ext.asyncio import AsyncSession
import logging

# Add backend to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database import async_session, engine
from app.models import Drug, DrugInteraction, TwosidesInteraction, Base
import app.diabetic.models
import app.prescription.models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def init_tables():
    logger.info("Initializing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Tables confirmed/created.")

def normalize_name(name):
    if pd.isna(name):
        return None
    return str(name).strip().upper()

async def ingest_twosides(csv_path: str):
    logger.info(f"Starting ingestion from {csv_path}")
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return

    chunksize = 200000
    total_processed = 0
    
    async with async_session() as session:
        # 1. Load existing drugs
        logger.info("Loading existing drugs...")
        result = await session.execute(select(Drug.id, Drug.name))
        drug_map = {row[1].upper(): row[0] for row in result.all()}
        logger.info(f"Loaded {len(drug_map)} existing drugs.")

        # 2. Keep track of seen interactions to avoid duplicates (drug1_id, drug2_id)
        # We only store unique pairs for DrugInteraction table
        seen_pairs = set()

        # Try to clear tables if needed
        # logger.info("Clearing existing TWOSIDES data...")
        # await session.execute(text("DELETE FROM twosides_interactions"))
        # await session.commit()

        logger.info("Pass: Processing chunks and inserting...")
        
        # Read the gzip file in chunks, skipping the first 5,000,000 rows already completed
        total_processed = 5000000
        for chunk in pd.read_csv(csv_path, compression='gzip', chunksize=chunksize, low_memory=False, skiprows=range(1, 5000001)):
            # Drop rows with missing essential data
            chunk = chunk.dropna(subset=['drug_1_concept_name', 'drug_2_concept_name', 'condition_concept_name'])
            
            # Extract unique new drugs in this chunk
            new_drugs_in_chunk = set()
            for name in chunk['drug_1_concept_name'].unique():
                norm = normalize_name(name)
                if norm and norm not in drug_map:
                    new_drugs_in_chunk.add(norm)
            for name in chunk['drug_2_concept_name'].unique():
                norm = normalize_name(name)
                if norm and norm not in drug_map:
                    new_drugs_in_chunk.add(norm)
            
            # Insert new drugs
            if new_drugs_in_chunk:
                drug_insert_data = [{"name": name, "is_approved": True} for name in new_drugs_in_chunk]
                # Insert and return ids to update drug_map
                for i in range(0, len(drug_insert_data), 1000):
                    batch = drug_insert_data[i:i+1000]
                    # Since sqlite might not support returning easy, we just commit then query
                    await session.execute(insert(Drug).values(batch))
                await session.commit()
                
                # Re-query the inserted drugs to update drug_map
                new_drugs_names = list(new_drugs_in_chunk)
                for i in range(0, len(new_drugs_names), 1000):
                    batch_names = new_drugs_names[i:i+1000]
                    res = await session.execute(select(Drug.id, Drug.name).where(func.upper(Drug.name).in_(batch_names)))
                    for row in res.all():
                        drug_map[row[1].upper()] = row[0]
            
            # Prepare interaction records
            twosides_records = []
            drug_interaction_records = []
            
            for _, row in chunk.iterrows():
                d1_name = normalize_name(row['drug_1_concept_name'])
                d2_name = normalize_name(row['drug_2_concept_name'])
                effect = str(row['condition_concept_name']).strip()
                try:
                    prr = float(row.get('PRR', 1.0))
                except (ValueError, TypeError):
                    prr = 1.0
                
                if not d1_name or not d2_name or not effect:
                    continue
                
                # 1. TwosidesInteraction (Raw logs)
                # twosides_records.append({
                #     "drug1_name": d1_name,
                #     "drug2_name": d2_name,
                #     "effect": effect,
                #     "source": "twosides",
                #     "severity": "moderate" if prr < 2.0 else "major"
                # })
                
                # 2. DrugInteraction (Summmary for API)
                id1 = drug_map.get(d1_name)
                id2 = drug_map.get(d2_name)
                
                if id1 and id2:
                    # ensure id1 < id2 for consistency
                    key = (min(id1, id2), max(id1, id2))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        drug_interaction_records.append({
                            "drug1_id": key[0],
                            "drug2_id": key[1],
                            "severity": "major" if prr >= 2.0 else "moderate",
                            "description": f"Potential interaction identified by TWOSIDES: {effect}",
                            "effect": effect,
                            "source": "twosides",
                            "confidence_score": 0.9 if prr >= 2.0 else 0.7
                        })
            
            # Insert DrugInteractions in bulk
            if drug_interaction_records:
                for i in range(0, len(drug_interaction_records), 5000):
                    batch = drug_interaction_records[i:i+5000]
                    await session.execute(insert(DrugInteraction).values(batch))
                await session.commit()
            
            # Note: We omit TwosidesInteraction raw logs to save space/time, as it would be billions of rows.
            
            total_processed += len(chunk)
            logger.info(f"Processed {total_processed} rows...")
            
            # Force garbage collection
            del chunk
            gc.collect()

    logger.info("Ingestion complete.")

async def main():
    await init_tables()
    data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'TWOSIDES.csv.gz')
    await ingest_twosides(data_file)

if __name__ == "__main__":
    asyncio.run(main())
