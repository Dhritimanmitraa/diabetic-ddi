"""
Load OffSIDES mined single-drug side-effect signals into the local database.

Downloads OFFSIDES.csv.gz from the Tatonetti Lab site, parses it, and stores
rows into the offsides_effects table with minimal columns:
- drug_name
- effect
- source (offsides)
- raw_row (JSON of the original row)

Usage:
    cd backend
    python -m scripts.load_offsides
"""
import asyncio
import gzip
import io
import json
import os
from typing import List

import pandas as pd
import requests
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session, init_db
from app.models import OffsidesEffect

OFFSIDES_URLS = [
    "https://tatonettilab.org/offsides/OFFSIDES.csv.gz",
    "https://tatonettilab.org/resources/offsides/OFFSIDES.csv.gz",
    "https://nsides.io/OFFSIDES.csv.gz",
    "https://raw.githubusercontent.com/TatonettiLab/offSIDES/master/OFFSIDES.csv.gz",
    "https://raw.githubusercontent.com/TatonettiLab/offSIDES/master/data/OFFSIDES.csv.gz",
]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "offsides")
os.makedirs(DATA_DIR, exist_ok=True)
# Accept either .gz or uncompressed .csv if provided manually
LOCAL_PATH_GZ = os.path.join(DATA_DIR, "OFFSIDES.csv.gz")
LOCAL_PATH_CSV = os.path.join(DATA_DIR, "OFFSIDES.csv")


async def load_offsides(session: AsyncSession, df: pd.DataFrame, chunk_size: int = 5000):
    inserted = 0
    col_names = [c.lower() for c in df.columns]
    drug_col = next((c for c in col_names if c in ["drug", "drug_name", "drug1", "drug_1", "drug_concept_name"]), None)
    effect_col = next((c for c in col_names if c in ["condition_concept_name"] or "effect" in c or "event" in c or "adr" in c or "side" in c), None)

    if not drug_col:
        raise ValueError(f"Could not infer drug column from OFFSIDES headers: {df.columns}")

    def get(row, col):
        return row[col] if col in row else None

    records: List[OffsidesEffect] = []
    for _, row in df.iterrows():
        d = str(get(row, drug_col)).strip()
        eff = str(get(row, effect_col)).strip() if effect_col else None
        if not d:
            continue
        records.append(
            OffsidesEffect(
                drug_name=d,
                effect=eff,
                source="offsides",
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


def download_offsides():
    if os.path.exists(LOCAL_PATH_GZ):
        return LOCAL_PATH_GZ
    if os.path.exists(LOCAL_PATH_CSV):
        return LOCAL_PATH_CSV
    last_err = None
    for url in OFFSIDES_URLS:
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            with open(LOCAL_PATH_GZ, "wb") as f:
                f.write(resp.content)
            return LOCAL_PATH_GZ
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to download OFFSIDES from all URLs. Last error: {last_err}. "
        f"If you have the file, place it at {LOCAL_PATH_GZ} or {LOCAL_PATH_CSV} and rerun."
    )


async def main():
    await init_db()

    path = download_offsides()
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            content = f.read()
        df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.read_csv(path)

    async with async_session() as session:
        inserted = await load_offsides(session, df)
        print(f"Inserted {inserted} OffSIDES rows.")


if __name__ == "__main__":
    asyncio.run(main())

