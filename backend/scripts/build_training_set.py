"""
Build labeled training set from TWOSIDES interactions.

Assembles drug pairs with severity labels for ML training:
1. Loads TWOSIDES interactions with mapped severities
2. Normalizes drug names to drugs table where possible
3. Extracts features using feature_engineering
4. Splits into train/val/test sets
5. Persists labeled CSV for reproducibility
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
import argparse
import sqlite3
import csv
import logging
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session, init_db
from app.models import Drug

# Severity label mapping to numeric
SEVERITY_TO_LABEL = {
    "fatal": 4,
    "contraindicated": 3,
    "major": 2,
    "moderate": 1,
    "minor": 0,
    None: -1,  # Unknown
}

LABEL_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_LABEL.items()}

DEFAULT_SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2}


async def load_drugs(db: AsyncSession) -> dict:
    """Load all drugs and create name lookup."""
    result = await db.execute(select(Drug))
    drugs = result.scalars().all()

    # Create multiple lookup keys (name, generic_name, lowercased)
    lookup = {}
    drug_data = {}

    for d in drugs:
        drug_dict = {
            "id": d.id,
            "name": d.name,
            "generic_name": d.generic_name,
            "drug_class": d.drug_class,
            "description": d.description,
            "mechanism": d.mechanism,
            "indication": d.indication,
            "molecular_weight": d.molecular_weight,
            "is_approved": d.is_approved,
        }
        drug_data[d.id] = drug_dict

        # Multiple lookup keys
        if d.name:
            lookup[d.name.lower().strip()] = d.id
        if d.generic_name:
            lookup[d.generic_name.lower().strip()] = d.id

    return lookup, drug_data


def normalize_drug_name(name: str, lookup: dict) -> int:
    """Try to match drug name to drugs table."""
    if not name:
        return None

    name_lower = name.lower().strip()

    # Direct match
    if name_lower in lookup:
        return lookup[name_lower]

    # Try removing common suffixes
    for suffix in [" hydrochloride", " hcl", " sodium", " potassium", " sulfate"]:
        cleaned = name_lower.replace(suffix, "").strip()
        if cleaned in lookup:
            return lookup[cleaned]

    # Try first word only
    first_word = name_lower.split()[0] if name_lower else ""
    if first_word in lookup:
        return lookup[first_word]

    return None


def sqlite_connect(db_path: Path) -> sqlite3.Connection:
    """Fast sqlite connection for large read/write workloads."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Speed pragmas (safe for local training jobs)
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("PRAGMA cache_size=-200000")  # ~200MB
    conn.commit()
    return conn


def iter_twosides_rows(
    conn: sqlite3.Connection,
    batch_size: int,
    max_rows: int | None,
):
    """Yield batches of TWOSIDES rows with severity using cursor-based pagination."""
    cur = conn.cursor()
    last_id = 0
    total = 0

    while True:
        if max_rows is not None and total >= max_rows:
            break
        limit = batch_size
        if max_rows is not None:
            limit = min(limit, max_rows - total)

        rows = cur.execute(
            """
            SELECT id, drug1_name, drug2_name, effect, severity, source
            FROM twosides_interactions
            WHERE id > ? AND severity IS NOT NULL
            ORDER BY id
            LIMIT ?
            """,
            (last_id, limit),
        ).fetchall()

        if not rows:
            break

        last_id = int(rows[-1][0])
        total += len(rows)
        yield rows, total


def parse_args():
    p = argparse.ArgumentParser(description="Build labeled training set from TWOSIDES")
    p.add_argument(
        "--max-positives",
        type=int,
        default=2_000_000,
        help="Cap TWOSIDES positives (0 = no cap)",
    )
    p.add_argument("--batch-size", type=int, default=50000, help="DB fetch batch size")
    p.add_argument(
        "--negatives",
        type=int,
        default=50000,
        help="Number of negative samples to generate",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "training"),
    )
    p.add_argument(
        "--write-full",
        action="store_true",
        help="Also write full_dataset.csv (can be large)",
    )
    return p.parse_args()


async def build_training_set():
    """Main function to build training set."""
    args = parse_args()
    print("=" * 60)
    print("Building Training Set from TWOSIDES")
    print("=" * 60)

    await init_db()

    # reduce noisy SQL logs
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    async with async_session() as db:
        # Load drugs
        print("\n1. Loading drugs from database...")
        drug_lookup, drug_data = await load_drugs(db)
        print(f"   Loaded {len(drug_data)} drugs, {len(drug_lookup)} lookup keys")

        # === STREAMING BUILD (no giant lists/dataframes) ===
        db_path = Path(__file__).parent.parent / "drug_interactions.db"
        conn = sqlite_connect(db_path)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "train.csv"
        val_path = out_dir / "val.csv"
        test_path = out_dir / "test.csv"
        full_path = out_dir / "full_dataset.csv"

        fieldnames = [
            "drug1_name",
            "drug2_name",
            "drug1_id",
            "drug2_id",
            "drug1_matched",
            "drug2_matched",
            "effect",
            "severity",
            "severity_label",
            "has_interaction",
            "source",
            "drug1_class",
            "drug1_mechanism",
            "drug2_class",
            "drug2_mechanism",
        ]

        rng = np.random.default_rng(args.seed)
        split_probs = DEFAULT_SPLIT

        def pick_split() -> str:
            r = rng.random()
            if r < split_probs["train"]:
                return "train"
            if r < split_probs["train"] + split_probs["val"]:
                return "val"
            return "test"

        print("\n2. Streaming TWOSIDES positives directly into train/val/test CSVs...")
        max_rows = None if args.max_positives == 0 else args.max_positives
        cap_msg = f" (capped at {max_rows:,})" if max_rows else " (no cap)"
        print(f"   batch_size={args.batch_size:,}{cap_msg}")

        severity_counts: dict[str, int] = {}
        pos_counts = {"train": 0, "val": 0, "test": 0}
        neg_counts = {"train": 0, "val": 0, "test": 0}
        matched = 0
        unmatched = 0

        # Writers
        f_train = open(train_path, "w", newline="", encoding="utf-8")
        f_val = open(val_path, "w", newline="", encoding="utf-8")
        f_test = open(test_path, "w", newline="", encoding="utf-8")
        writers = {
            "train": csv.DictWriter(f_train, fieldnames=fieldnames),
            "val": csv.DictWriter(f_val, fieldnames=fieldnames),
            "test": csv.DictWriter(f_test, fieldnames=fieldnames),
        }
        for w in writers.values():
            w.writeheader()

        full_writer = None
        f_full = None
        if args.write_full:
            f_full = open(full_path, "w", newline="", encoding="utf-8")
            full_writer = csv.DictWriter(f_full, fieldnames=fieldnames)
            full_writer.writeheader()

        try:
            for rows, total_loaded in iter_twosides_rows(
                conn, args.batch_size, max_rows
            ):
                for _id, d1n, d2n, effect, severity, source in rows:
                    sev = severity or "minor"
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1

                    drug1_id = normalize_drug_name(d1n, drug_lookup)
                    drug2_id = normalize_drug_name(d2n, drug_lookup)

                    if drug1_id:
                        matched += 1
                    else:
                        unmatched += 1

                    rec = {
                        "drug1_name": d1n,
                        "drug2_name": d2n,
                        "drug1_id": drug1_id,
                        "drug2_id": drug2_id,
                        "drug1_matched": int(drug1_id is not None),
                        "drug2_matched": int(drug2_id is not None),
                        "effect": effect,
                        "severity": sev,
                        "severity_label": SEVERITY_TO_LABEL.get(sev, -1),
                        "has_interaction": 1,
                        "source": source or "twosides",
                        "drug1_class": (
                            drug_data.get(drug1_id, {}).get("drug_class")
                            if drug1_id
                            else None
                        ),
                        "drug1_mechanism": (
                            drug_data.get(drug1_id, {}).get("mechanism")
                            if drug1_id
                            else None
                        ),
                        "drug2_class": (
                            drug_data.get(drug2_id, {}).get("drug_class")
                            if drug2_id
                            else None
                        ),
                        "drug2_mechanism": (
                            drug_data.get(drug2_id, {}).get("mechanism")
                            if drug2_id
                            else None
                        ),
                    }

                    s = pick_split()
                    writers[s].writerow(rec)
                    pos_counts[s] += 1
                    if full_writer:
                        full_writer.writerow(rec)

                if total_loaded % 500_000 == 0:
                    print(f"   Streamed positives: {total_loaded:,}")

            print("\n3. Generating negatives (streaming)...")
            drug_names = list({d["name"] for d in drug_data.values() if d.get("name")})
            n_negative = int(args.negatives)
            attempts = 0
            max_attempts = n_negative * 20
            negative_pairs: set[tuple[str, str]] = set()

            while sum(neg_counts.values()) < n_negative and attempts < max_attempts:
                attempts += 1
                d1_name = rng.choice(drug_names)
                d2_name = rng.choice(drug_names)
                if d1_name == d2_name:
                    continue
                pair = (d1_name.lower(), d2_name.lower())
                if pair in negative_pairs:
                    continue

                negative_pairs.add(pair)

                d1_id = drug_lookup.get(d1_name.lower())
                d2_id = drug_lookup.get(d2_name.lower())
                rec = {
                    "drug1_name": d1_name,
                    "drug2_name": d2_name,
                    "drug1_id": d1_id,
                    "drug2_id": d2_id,
                    "drug1_matched": int(d1_id is not None),
                    "drug2_matched": int(d2_id is not None),
                    "effect": None,
                    "severity": "none",
                    "severity_label": -1,
                    "has_interaction": 0,
                    "source": "negative_sample",
                    "drug1_class": (
                        drug_data.get(d1_id, {}).get("drug_class") if d1_id else None
                    ),
                    "drug1_mechanism": (
                        drug_data.get(d1_id, {}).get("mechanism") if d1_id else None
                    ),
                    "drug2_class": (
                        drug_data.get(d2_id, {}).get("drug_class") if d2_id else None
                    ),
                    "drug2_mechanism": (
                        drug_data.get(d2_id, {}).get("mechanism") if d2_id else None
                    ),
                }
                s = pick_split()
                writers[s].writerow(rec)
                neg_counts[s] += 1
                if full_writer:
                    full_writer.writerow(rec)

            if sum(neg_counts.values()) < n_negative:
                print(
                    f"   Warning: generated only {sum(neg_counts.values()):,} negatives (attempts={attempts:,})"
                )
            else:
                print(
                    f"   Generated {sum(neg_counts.values()):,} negatives (attempts={attempts:,})"
                )

        finally:
            f_train.close()
            f_val.close()
            f_test.close()
            if f_full:
                f_full.close()
            conn.close()

        total_samples = sum(pos_counts.values()) + sum(neg_counts.values())
        print("\n4. Dataset summary:")
        print(
            f"   Positives: {sum(pos_counts.values()):,}  Negatives: {sum(neg_counts.values()):,}  Total: {total_samples:,}"
        )
        print(
            f"   Train: {pos_counts['train'] + neg_counts['train']:,} | Val: {pos_counts['val'] + neg_counts['val']:,} | Test: {pos_counts['test'] + neg_counts['test']:,}"
        )

        # Save metadata
        metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "max_positives": max_rows,
            "batch_size": args.batch_size,
            "negatives": int(args.negatives),
            "total_samples": int(total_samples),
            "train_samples": int(pos_counts["train"] + neg_counts["train"]),
            "val_samples": int(pos_counts["val"] + neg_counts["val"]),
            "test_samples": int(pos_counts["test"] + neg_counts["test"]),
            "positive_samples": int(sum(pos_counts.values())),
            "negative_samples": int(sum(neg_counts.values())),
            "severity_distribution": severity_counts,
            "matched_to_drugs_table_drug1_only": int(matched),
            "unmatched_to_drugs_table_drug1_only": int(unmatched),
            "source": "TWOSIDES",
            "split": split_probs,
            "write_full_dataset": bool(args.write_full),
        }

        import json

        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n   Saved to {out_dir}")
        print("\n" + "=" * 60)
        print("Training set build complete!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(build_training_set())
