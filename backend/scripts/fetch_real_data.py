"""
Fetch REAL drug interaction data from public APIs.

Data Sources:
1. OpenFDA - FDA Adverse Event Reporting System (https://open.fda.gov)
2. RxNorm/NIH - National Institutes of Health (https://rxnav.nlm.nih.gov)

These are REAL government databases with millions of drug records and interactions.

Usage:
    python -m scripts.fetch_real_data

    Or with custom limits:
    python -m scripts.fetch_real_data --drugs 10000 --interactions 100000
"""

import asyncio
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import engine, async_session, init_db
from app.models import Drug, DrugInteraction
from app.services.data_fetcher import DrugDataFetcher
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def load_drugs_to_db(db: AsyncSession, drugs: list) -> dict:
    """Load fetched drugs into database."""
    drug_map = {}
    added = 0
    skipped = 0

    for drug_data in drugs:
        name = drug_data.get("name", "").strip()
        if not name:
            continue

        # Check if drug already exists
        existing = await db.execute(
            select(Drug).where(func.upper(Drug.name) == name.upper())
        )
        if existing.scalar_one_or_none():
            skipped += 1
            continue

        drug = Drug(
            name=name,
            generic_name=drug_data.get("generic_name"),
            drug_class=drug_data.get("drug_class"),
            description=drug_data.get("description"),
            indication=drug_data.get("indication"),
            mechanism=drug_data.get("mechanism"),
            drugbank_id=drug_data.get("drugbank_id"),
            is_approved=True,
        )
        db.add(drug)
        await db.flush()
        drug_map[name.upper()] = drug.id
        added += 1

    await db.commit()
    logger.info(f"Added {added} new drugs, skipped {skipped} existing")

    # Get complete drug map
    all_drugs = await db.execute(select(Drug))
    for drug in all_drugs.scalars():
        drug_map[drug.name.upper()] = drug.id

    return drug_map


async def load_interactions_to_db(
    db: AsyncSession, interactions: list, drug_map: dict
) -> int:
    """Load fetched interactions into database."""
    added = 0
    skipped = 0

    for interaction_data in interactions:
        drug1_name = interaction_data.get("drug1_name", "").upper()
        drug2_name = interaction_data.get("drug2_name", "").upper()

        drug1_id = drug_map.get(drug1_name)
        drug2_id = drug_map.get(drug2_name)

        if not drug1_id or not drug2_id:
            skipped += 1
            continue

        # Check if interaction exists
        existing = await db.execute(
            select(DrugInteraction).where(
                (
                    (DrugInteraction.drug1_id == drug1_id)
                    & (DrugInteraction.drug2_id == drug2_id)
                )
                | (
                    (DrugInteraction.drug1_id == drug2_id)
                    & (DrugInteraction.drug2_id == drug1_id)
                )
            )
        )
        if existing.scalar_one_or_none():
            skipped += 1
            continue

        interaction = DrugInteraction(
            drug1_id=drug1_id,
            drug2_id=drug2_id,
            severity=interaction_data.get("severity", "moderate"),
            description=interaction_data.get("description"),
            effect=interaction_data.get("effect"),
            mechanism=interaction_data.get("mechanism"),
            management=interaction_data.get("management"),
            source=interaction_data.get("source", "api"),
            evidence_level=interaction_data.get("evidence_level", "case_report"),
            confidence_score=interaction_data.get("confidence_score", 0.7),
        )
        db.add(interaction)
        added += 1

        # Commit in batches
        if added % 1000 == 0:
            await db.commit()
            logger.info(f"Committed {added} interactions...")

    await db.commit()
    logger.info(f"Added {added} new interactions, skipped {skipped}")
    return added


async def fetch_and_load(target_drugs: int = 5000, target_interactions: int = 100000):
    """Main function to fetch real data and load into database."""

    print("=" * 70)
    print("  FETCHING REAL DRUG DATA FROM PUBLIC APIs")
    print("=" * 70)
    print()
    print("  Data Sources:")
    print("  • OpenFDA - FDA Adverse Event Reporting System")
    print("  • RxNorm  - NIH Drug Database")
    print()
    print(f"  Target: {target_drugs:,} drugs, {target_interactions:,} interactions")
    print("=" * 70)
    print()

    # Initialize database
    logger.info("Initializing database...")
    await init_db()

    # Create fetcher
    fetcher = DrugDataFetcher(data_dir="./data")

    # Fetch data from APIs
    logger.info("Fetching data from OpenFDA and RxNorm APIs...")
    logger.info("This may take several minutes depending on network speed...")

    data = await fetcher.fetch_comprehensive_data(
        target_drugs=target_drugs, target_interactions=target_interactions
    )

    # Save raw data to file
    fetcher.save_data(data, "real_drug_data.json")

    # Load into database
    async with async_session() as db:
        logger.info("Loading drugs into database...")
        drug_map = await load_drugs_to_db(db, data["drugs"])

        logger.info("Loading interactions into database...")
        await load_interactions_to_db(db, data["interactions"], drug_map)

        # Get final counts
        drug_count = await db.execute(select(func.count(Drug.id)))
        interaction_count = await db.execute(select(func.count(DrugInteraction.id)))

        print()
        print("=" * 70)
        print("  DATA FETCH COMPLETE!")
        print("=" * 70)
        print(f"  Total drugs in database:       {drug_count.scalar():,}")
        print(f"  Total interactions in database: {interaction_count.scalar():,}")
        print()
        print("  Data sources:")
        print("  • OpenFDA: https://open.fda.gov")
        print("  • RxNorm:  https://rxnav.nlm.nih.gov")
        print("=" * 70)


def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Fetch real drug interaction data from FDA and NIH APIs"
    )
    parser.add_argument(
        "--drugs",
        type=int,
        default=5000,
        help="Target number of drugs to fetch (default: 5000)",
    )
    parser.add_argument(
        "--interactions",
        type=int,
        default=100000,
        help="Target number of interactions to fetch (default: 100000)",
    )

    args = parser.parse_args()

    asyncio.run(
        fetch_and_load(target_drugs=args.drugs, target_interactions=args.interactions)
    )


if __name__ == "__main__":
    main()
