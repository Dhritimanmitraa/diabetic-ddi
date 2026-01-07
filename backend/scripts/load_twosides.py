"""
Load TWOSIDES/OffSIDES mined DDI data into the local database.

Downloads TWOSIDES.csv.gz from the Tatonetti Lab site, parses it, and stores
rows into the twosides_interactions table with minimal columns:
- drug1_name
- drug2_name
- effect (if available)
- source (twosides)
- raw_row (JSON of the original row for future enrichment)

Usage:
    python -m scripts.load_twosides
"""

# pyright: reportMissingImports=false
import asyncio
import gzip
import io
import json
import os
from typing import List

import pandas as pd
import requests  # type: ignore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session, init_db
from app.models import TwosidesInteraction

TWOSIDES_URL = "https://tatonettilab.org/offsides/TWOSIDES.csv.gz"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "twosides")
os.makedirs(DATA_DIR, exist_ok=True)
LOCAL_PATH_GZ = os.path.join(DATA_DIR, "TWOSIDES.csv.gz")
LOCAL_PATH_CSV = os.path.join(DATA_DIR, "TWOSIDES.csv")


async def load_twosides(
    session: AsyncSession, df: pd.DataFrame, chunk_size: int = 5000
):
    inserted = 0
    # Expected columns in TWOSIDES: drug1, drug2, side_effect_name or event_name; this can vary.
    # Common real-world headers: drug_1_concept_name, drug_2_concept_name, condition_concept_name
    col_names = [c.lower() for c in df.columns]

    # Heuristic column picks (ordered by preference)
    drug1_candidates = [
        "drug1",
        "drug_1",
        "drug_a",
        "drug",
        "drug 1",
        "drug_1_concept_name",
        "drug_1_name",
    ]
    drug2_candidates = [
        "drug2",
        "drug_2",
        "drug_b",
        "drug2",
        "drug 2",
        "drug_2_concept_name",
        "drug_2_name",
    ]
    effect_candidates = [
        "side_effect_name",
        "event_name",
        "effect",
        "adr",
        "condition_concept_name",
        "condition_name",
    ]

    drug1_col = next((c for c in col_names if c in drug1_candidates), None)
    drug2_col = next((c for c in col_names if c in drug2_candidates), None)
    effect_col = next(
        (
            c
            for c in col_names
            if c in effect_candidates or "effect" in c or "event" in c or "adr" in c
        ),
        None,
    )

    if not drug1_col or not drug2_col:
        raise ValueError(
            f"Could not infer drug columns from TWOSIDES headers: {df.columns}"
        )

    # Normalize column access
    def get(row, col):
        return row[col] if col in row else None

    records: List[TwosidesInteraction] = []
    for _, row in df.iterrows():
        d1 = str(get(row, drug1_col)).strip()
        d2 = str(get(row, drug2_col)).strip()
        eff = str(get(row, effect_col)).strip() if effect_col else None
        if not d1 or not d2:
            continue
        records.append(
            TwosidesInteraction(
                drug1_name=d1,
                drug2_name=d2,
                effect=eff,
                source="twosides",
                raw_row=json.dumps(row.to_dict(), default=str),
            )
        )
        if len(records) >= chunk_size:
            session.add_all(records)
            await session.commit()
            inserted += len(records)
            records.clear()

    if records:
        session.add_all(records)
        await session.commit()
        inserted += len(records)

    return inserted


def get_local_twosides_path():
    """
    Resolve TWOSIDES local path.
    Preference:
    - If TWOSIDES.csv.gz exists in data/twosides, use it.
    - Else if TWOSIDES.csv exists, use it (uncompressed).
    - Else download the .gz from remote.
    """
    if os.path.exists(LOCAL_PATH_GZ):
        return LOCAL_PATH_GZ, "gz"
    if os.path.exists(LOCAL_PATH_CSV):
        return LOCAL_PATH_CSV, "csv"
    resp = requests.get(TWOSIDES_URL, timeout=120)
    resp.raise_for_status()
    with open(LOCAL_PATH_GZ, "wb") as f:
        f.write(resp.content)
    return LOCAL_PATH_GZ, "gz"


async def main():
    await init_db()

    path, kind = get_local_twosides_path()

    # Stream chunks to avoid loading full CSV into memory
    reader = pd.read_csv(
        path if kind == "csv" else path,
        chunksize=50000,
        compression="gzip" if kind == "gz" else None,
    )

    total_inserted = 0
    async with async_session() as session:
        async for chunk in _iter_chunks(reader):
            inserted = await load_twosides(session, chunk)
            total_inserted += inserted

    print(f"Inserted {total_inserted} TWOSIDES rows.")


async def _iter_chunks(reader):
    # pandas TextFileReader is synchronous; wrap in async generator
    for chunk in reader:
        yield chunk


if __name__ == "__main__":
    asyncio.run(main())
