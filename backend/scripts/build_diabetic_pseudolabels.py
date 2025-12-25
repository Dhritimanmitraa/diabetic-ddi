"""
Build pseudo-labeled diabetic drug risk dataset from MIMIC-IV demo data.

Pipeline:
1) Load MIMIC-IV demo (hosp + icu) tables from data/mimiciv/demo/...
2) Derive patient context (age, sex, comorbid flags, select labs).
3) For each prescription, apply diabetic rules engine to generate a pseudo-label.
4) Split into train/val/test and persist CSVs for model training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# Local imports
from app.diabetic.rules import DiabeticDrugRules
from sklearn.model_selection import train_test_split

# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).parent.parent
MIMIC_DIR = (
    BASE_DIR / "data" / "mimiciv" / "demo" / "mimic-iv-clinical-database-demo-2.2"
)
OUTPUT_DIR = BASE_DIR / "data" / "diabetic" / "training"

# Map risk levels to integers for downstream ML training
RISK_LEVEL_TO_LABEL = {
    "safe": 0,
    "caution": 1,
    "high_risk": 2,
    "contraindicated": 3,
    "fatal": 4,
}

# Lab ITEMID mappings (common IDs in MIMIC-IV; best-effort for demo subset)
LAB_ITEM_MAP = {
    "creatinine": [50912, 1525, 220615],
    "potassium": [50971, 227442],
    "glucose": [50931, 220621],
}

# ICD code prefix helpers for comorbidities (ICD-9/10)
ICD_PREFIXES = {
    "diabetes": ["E10", "E11", "E13", "250"],
    "nephropathy": ["N18", "N17", "585", "584"],
    "retinopathy": ["E1131", "E1132", "3620"],
    "neuropathy": ["E1140", "E1142", "3572"],
    "cardiovascular": ["I50", "I25", "I21", "410", "411", "412", "413", "414"],
    "hypertension": ["I10", "I11", "I12", "I13", "401", "402", "403", "404"],
    "hyperlipidemia": ["E78", "272"],
    "obesity": ["E66", "2780"],
}


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_table(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    kwargs: dict[str, Any] = {"compression": "gzip"}
    if columns is not None:
        kwargs["usecols"] = columns
    return pd.read_csv(path, **kwargs)  # type: ignore[arg-type]


def load_patients() -> pd.DataFrame:
    patients_path = MIMIC_DIR / "hosp" / "patients.csv.gz"
    cols = ["subject_id", "anchor_age", "gender"]
    df = load_table(patients_path, cols)
    df = df.rename(columns={"anchor_age": "age"})
    return df


def load_diagnoses() -> pd.DataFrame:
    diag_path = MIMIC_DIR / "hosp" / "diagnoses_icd.csv.gz"
    cols = ["subject_id", "icd_code"]
    return load_table(diag_path, cols)


def load_labs() -> pd.DataFrame:
    labs_path = MIMIC_DIR / "hosp" / "labevents.csv.gz"
    cols = ["subject_id", "itemid", "valuenum", "charttime"]
    return load_table(labs_path, cols)


def load_prescriptions() -> pd.DataFrame:
    rx_path = MIMIC_DIR / "hosp" / "prescriptions.csv.gz"
    # MIMIC demo columns (no drug_name_poe/generic in demo)
    cols = [
        "subject_id",
        "drug",
        "formulary_drug_cd",
        "gsn",
        "ndc",
        "starttime",
        "route",
    ]
    return load_table(rx_path, cols)


def has_prefix(code: str, prefixes: List[str]) -> bool:
    if not isinstance(code, str):
        return False
    code_u = code.upper()
    return any(code_u.startswith(p.upper()) for p in prefixes)


def build_comorbidity_flags(diagnoses: pd.DataFrame) -> Dict[int, Dict]:
    flags: Dict[int, Dict] = {}
    grouped = diagnoses.groupby("subject_id")["icd_code"].apply(list)
    for sid, codes in grouped.items():
        flags[int(str(sid))] = {  # type: ignore
            "has_nephropathy": any(
                has_prefix(c, ICD_PREFIXES["nephropathy"]) for c in codes
            ),
            "has_retinopathy": any(
                has_prefix(c, ICD_PREFIXES["retinopathy"]) for c in codes
            ),
            "has_neuropathy": any(
                has_prefix(c, ICD_PREFIXES["neuropathy"]) for c in codes
            ),
            "has_cardiovascular": any(
                has_prefix(c, ICD_PREFIXES["cardiovascular"]) for c in codes
            ),
            "has_hypertension": any(
                has_prefix(c, ICD_PREFIXES["hypertension"]) for c in codes
            ),
            "has_hyperlipidemia": any(
                has_prefix(c, ICD_PREFIXES["hyperlipidemia"]) for c in codes
            ),
            "has_obesity": any(has_prefix(c, ICD_PREFIXES["obesity"]) for c in codes),
        }
    return flags


def build_lab_features(labs: pd.DataFrame) -> Dict[int, Dict]:
    feats: Dict[int, Dict] = {}
    if labs.empty:
        return feats

    labs = labs.dropna(subset=["valuenum"])

    for lab_name, itemids in LAB_ITEM_MAP.items():
        subset = labs[labs["itemid"].isin(itemids)]
        if subset.empty:
            continue
        latest = subset.sort_values("charttime").groupby("subject_id").tail(1)  # type: ignore
        for _, row in latest.iterrows():
            sid = int(row["subject_id"])
            feats.setdefault(sid, {})
            feats[sid][lab_name] = float(row["valuenum"])
    return feats


def build_patient_contexts() -> Dict[int, Dict]:
    patients = load_patients()
    diagnoses = load_diagnoses()
    labs = load_labs()

    comorb = build_comorbidity_flags(diagnoses)
    lab_feats = build_lab_features(labs)

    contexts: Dict[int, Dict] = {}
    for _, row in patients.iterrows():
        sid = int(row["subject_id"])
        age_val = row["age"]
        gender_val = row["gender"]
        # When using iterrows(), values are scalars
        ctx = {
            "age": float(age_val) if age_val is not None and not (isinstance(age_val, float) and pd.isna(age_val)) else None,  # type: ignore
            "gender": str(gender_val).lower() if gender_val is not None and not (isinstance(gender_val, float) and pd.isna(gender_val)) else None,  # type: ignore
            "diabetes_type": "type_2",  # default; demo data lacks explicit type
            "years_with_diabetes": None,
            "hba1c": None,
            "fasting_glucose": None,
            "egfr": None,  # not computed from demo
            "creatinine": None,
            "potassium": None,
            "alt": None,
            "ast": None,
            "has_nephropathy": False,
            "has_retinopathy": False,
            "has_neuropathy": False,
            "has_cardiovascular": False,
            "has_hypertension": False,
            "has_hyperlipidemia": False,
            "has_obesity": False,
            "bmi": None,
        }

        if sid in comorb:
            ctx.update(comorb[sid])
        if sid in lab_feats:
            lf = lab_feats[sid]
            ctx["creatinine"] = lf.get("creatinine")
            ctx["potassium"] = lf.get("potassium")
            # Optionally map glucose to fasting_glucose if present
            ctx["fasting_glucose"] = lf.get("glucose")

        contexts[sid] = ctx

    return contexts


def choose_drug_name(row: pd.Series) -> str | None:
    for col in ["drug", "formulary_drug_cd", "gsn", "ndc"]:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None  # type: ignore[return-value]


def generate_pseudolabels() -> pd.DataFrame:
    print("Loading patient contexts...")
    patient_ctx = build_patient_contexts()

    print("Loading prescriptions...")
    rx = load_prescriptions()
    rx["drug_clean"] = rx.apply(choose_drug_name, axis=1)
    rx = rx.dropna(subset=["drug_clean"])

    rules = DiabeticDrugRules()
    records: List[Dict] = []

    # Pre-build patient med lists to capture within-patient combos
    med_lists: Dict[int, List[str]] = {}
    for sid, group in rx.groupby("subject_id"):
        meds = [
            m for m in group["drug_clean"].str.lower().tolist() if isinstance(m, str)
        ]
        med_lists[int(str(sid))] = meds  # type: ignore

    print("Generating pseudo-labels from rules...")
    for sid, group in rx.groupby("subject_id"):
        sid_int = int(str(sid))  # type: ignore
        ctx = patient_ctx.get(sid_int)
        if not ctx:
            continue
        current_meds = med_lists.get(sid_int, [])

        for _, row in group.iterrows():
            drug = str(row["drug_clean"])  # type: ignore
            assessment = rules.assess_drug_risk(drug, ctx, current_meds)

            records.append(
                {
                    "subject_id": sid_int,
                    "drug_name": drug,
                    "age": ctx.get("age"),
                    "gender": ctx.get("gender"),
                    "creatinine": ctx.get("creatinine"),
                    "potassium": ctx.get("potassium"),
                    "fasting_glucose": ctx.get("fasting_glucose"),
                    "has_nephropathy": ctx.get("has_nephropathy"),
                    "has_retinopathy": ctx.get("has_retinopathy"),
                    "has_neuropathy": ctx.get("has_neuropathy"),
                    "has_cardiovascular": ctx.get("has_cardiovascular"),
                    "has_hypertension": ctx.get("has_hypertension"),
                    "has_hyperlipidemia": ctx.get("has_hyperlipidemia"),
                    "has_obesity": ctx.get("has_obesity"),
                    "risk_level": assessment.risk_level,
                    "severity": assessment.severity,
                    "risk_score": assessment.risk_score,
                    "rule_references": "|".join(assessment.rule_references),
                    "evidence_sources": "|".join(assessment.evidence_sources),
                    "patient_factors": "|".join(assessment.patient_factors),
                    "monitoring": "|".join(assessment.monitoring),
                    "alternatives": "|".join(assessment.alternatives),
                    "label": RISK_LEVEL_TO_LABEL.get(assessment.risk_level, 0),
                }
            )

    df = pd.DataFrame(records)
    return df


def save_splits(df: pd.DataFrame):
    ensure_dirs()
    if df.empty:
        raise RuntimeError("No data generated from pseudo-labeling.")

    stratify_labels = df["label"] if df["label"].nunique() > 1 else None  # type: ignore
    train_indices, temp_indices = train_test_split(
        df.index, test_size=0.3, random_state=42, stratify=stratify_labels
    )
    train = df.loc[train_indices]
    temp = df.loc[temp_indices]
    
    temp_strat = temp["label"] if temp["label"].nunique() > 1 else None  # type: ignore
    val_indices, test_indices = train_test_split(
        temp.index, test_size=0.5, random_state=42, stratify=temp_strat
    )
    val = temp.loc[val_indices]
    test = temp.loc[test_indices]

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    df.to_csv(OUTPUT_DIR / "full_dataset.csv", index=False)

    metadata = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "total_rows": len(df),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "risk_distribution": df["risk_level"].value_counts().to_dict(),
        "source": "mimic_demo_rules_pseudo",
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved splits to", OUTPUT_DIR)
    print("Distribution:", metadata["risk_distribution"])


def main():
    ensure_dirs()
    df = generate_pseudolabels()
    print(f"Generated {len(df)} pseudo-labeled rows")
    save_splits(df)
    print("Done.")


if __name__ == "__main__":
    main()
