"""
Map OffSIDES effects to coarse severities and store them.

Heuristic mapping (same idea as TWOSIDES mapping):
- fatal: keywords like death, fatal, anaphylaxis, cardiac arrest
- major: hospitalization, shock, bleeding/hemorrhage, lactic acidosis, renal failure, arrhythmia, seizure, stroke
- moderate: syncope, hypotension, hyperkalemia, hypoglycemia, pancreatitis, severe nausea/vomiting
- minor: default fallback

Run:
    cd backend
    python -m scripts.map_offsides_severity
"""
import asyncio
import json
import sqlite3
import os
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session, init_db
from app.models import OffsidesEffect


def classify_effect(effect: str) -> str:
    if not effect:
        return "minor"
    e = effect.lower()
    fatal_kw = ["death", "fatal", "cardiac arrest", "anaphylaxis"]
    major_kw = [
        "hospital", "hospitalization", "shock", "hemorrhage", "bleeding",
        "lactic acidosis", "renal failure", "kidney failure", "arrhythmia",
        "seizure", "stroke", "torsades", "ventricular", "mi", "myocardial infarction",
        "gi bleed", "internal bleed"
    ]
    moderate_kw = [
        "syncope", "hypotension", "hyperkalemia", "hypoglycemia",
        "pancreatitis", "severe nausea", "severe vomiting", "ak i", "aki"
    ]
    if any(k in e for k in fatal_kw):
        return "fatal"
    if any(k in e for k in major_kw):
        return "major"
    if any(k in e for k in moderate_kw):
        return "moderate"
    return "minor"


async def map_severity(session: AsyncSession, batch: int = 5000):
    offset = 0
    updated = 0
    while True:
        result = await session.execute(
            select(OffsidesEffect).where(OffsidesEffect.severity == None).offset(offset).limit(batch)
        )
        rows = result.scalars().all()
        if not rows:
            break
        for r in rows:
            eff = r.effect
            if not eff and r.raw_row:
                try:
                    data = json.loads(r.raw_row)
                    eff = data.get("condition_concept_name") or data.get("effect")
                except Exception:
                    eff = None
            r.severity = classify_effect(eff)
        await session.commit()
        updated += len(rows)
        offset += batch
    return updated


def ensure_severity_column(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(offsides_effects)")
    cols = [r[1] for r in cur.fetchall()]
    if "severity" not in cols:
        cur.execute("ALTER TABLE offsides_effects ADD COLUMN severity TEXT")
        conn.commit()
    conn.close()


async def main():
    await init_db()
    db_path = os.path.join(os.path.dirname(__file__), "..", "drug_interactions.db")
    ensure_severity_column(os.path.abspath(db_path))
    async with async_session() as session:
        updated = await map_severity(session)
        print(f"Updated severity for {updated} OffSIDES rows.")


if __name__ == "__main__":
    asyncio.run(main())

