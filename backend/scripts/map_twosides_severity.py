"""
Map TWOSIDES/OffSIDES effects to coarse severities and store them.

Heuristic mapping:
- fatal/contra: keywords like death, fatal, anaphylaxis, cardiac arrest
- major: hospitalization, shock, bleeding/hemorrhage, lactic acidosis, renal failure, arrhythmia, seizure
- moderate: syncope, hypotension, hypoglycemia, hyperkalemia, pancreatitis, severe nausea/vomiting
- minor: default fallback

This script:
1) Ensures the severity column exists on twosides_interactions (SQLite ALTER TABLE if needed).
2) Scans rows and assigns severity if missing.

Run:
    cd backend
    python -m scripts.map_twosides_severity
"""
import asyncio
import logging
import json
import sqlite3
import os
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session, init_db
from app.models import TwosidesInteraction

logger = logging.getLogger(__name__)


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
    # Use cursor-based pagination (WHERE id > last_id) instead of OFFSET for speed
    last_id = 0
    updated = 0
    total_processed = 0
    
    while True:
        # Fetch rows missing severity using cursor-based pagination
        result = await session.execute(
            text("""
                SELECT id, effect, raw_row 
                FROM twosides_interactions 
                WHERE id > :last_id AND severity IS NULL 
                ORDER BY id 
                LIMIT :batch
            """),
            {"last_id": last_id, "batch": batch}
        )
        rows = result.fetchall()
        
        if not rows:
            break
        
        # Prepare batch updates
        updates = []
        for row in rows:
            row_id, effect, raw_row = row
            # fallback: try raw_row if effect missing
            if not effect and raw_row:
                try:
                    data = json.loads(raw_row)
                    effect = data.get("EVENT", data.get("effect", None))
                except Exception:
                    effect = None
            severity = classify_effect(effect if effect else "")
            updates.append((severity, row_id))
            last_id = row_id  # Track last processed ID
        
        # Batch update - split into sub-batches to avoid huge IN clauses
        if updates:
            # Process in sub-batches of 500 to avoid SQLite query size limits
            sub_batch_size = 500
            for i in range(0, len(updates), sub_batch_size):
                sub_batch = updates[i:i + sub_batch_size]
                
                # Build CASE statement for this sub-batch
                case_whens = []
                ids_list = []
                for sev, rid in sub_batch:
                    sev_escaped = sev.replace("'", "''")
                    case_whens.append(f"WHEN {rid} THEN '{sev_escaped}'")
                    ids_list.append(str(rid))
                
                case_sql = " ".join(case_whens)
                ids_sql = ",".join(ids_list)
                
                # Update this sub-batch
                batch_update_sql = f"""
                    UPDATE twosides_interactions 
                    SET severity = CASE id 
                        {case_sql}
                    END
                    WHERE id IN ({ids_sql})
                """
                await session.execute(text(batch_update_sql))
            
            await session.commit()
            updated += len(updates)
            total_processed += len(updates)
            
            # Progress indicator
            if total_processed % 50000 == 0:
                print(f"Processed {total_processed} rows, last_id={last_id}...")
    
    return updated


def ensure_severity_column(db_path: str):
    # SQLite: add severity column and index if missing
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(twosides_interactions)")
    cols = [r[1] for r in cur.fetchall()]
    if "severity" not in cols:
        cur.execute("ALTER TABLE twosides_interactions ADD COLUMN severity TEXT")
        conn.commit()
    
    # Create index on severity for faster WHERE severity IS NULL queries
    cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_twosides_severity'")
    if not cur.fetchone():
        print("Creating index on severity column...")
        cur.execute("CREATE INDEX idx_twosides_severity ON twosides_interactions(severity)")
        conn.commit()
        print("Index created.")
    
    conn.close()


async def main():
    await init_db()
    db_path = os.path.join(os.path.dirname(__file__), "..", "drug_interactions.db")
    ensure_severity_column(os.path.abspath(db_path))
    print("Starting optimized severity mapping (cursor-based pagination)...")
    async with async_session() as session:
        updated = await map_severity(session)
        print(f"\nUpdated severity for {updated} TWOSIDES rows.")


if __name__ == "__main__":
    asyncio.run(main())

