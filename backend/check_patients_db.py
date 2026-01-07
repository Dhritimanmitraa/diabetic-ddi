"""Quick script to check patients in database."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.database import async_session
from app.diabetic.models import DiabeticPatient
from sqlalchemy import select


async def check_patients():
    async with async_session() as db:
        result = await db.execute(
            select(
                DiabeticPatient.patient_id,
                DiabeticPatient.name,
                DiabeticPatient.diabetes_type,
            )
        )
        patients = result.all()

        print(f"\n{'='*60}")
        print(f"Patients in Database: {len(patients)}")
        print(f"{'='*60}")
        for p in patients:
            print(f"  {p[0]}: {p[1]} ({p[2]})")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(check_patients())
