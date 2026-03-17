"""
Medication Adherence Tracking Router.

Provides endpoints for managing medication schedules and logging
dose taken/missed events. Supports per-user schedule management
and adherence statistics.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import AdherenceLog, MedicationSchedule, User
from app.services.jwt_auth import require_current_user

router = APIRouter(prefix="/adherence", tags=["Adherence"])
logger = logging.getLogger(__name__)


# ── Schemas ──────────────────────────────────────────────────────────────

class ScheduleCreate(BaseModel):
    drug_name: str = Field(..., min_length=1, max_length=255)
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    time_of_day: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    notes: Optional[str] = None


class ScheduleOut(BaseModel):
    id: int
    user_id: str
    drug_name: str
    dosage: Optional[str]
    frequency: Optional[str]
    time_of_day: Optional[str]
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    notes: Optional[str]
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class LogCreate(BaseModel):
    schedule_id: int
    status: str = Field(default="taken", pattern=r"^(taken|missed|skipped)$")
    notes: Optional[str] = None


class LogOut(BaseModel):
    id: int
    schedule_id: int
    status: str
    logged_at: datetime
    notes: Optional[str]

    model_config = {"from_attributes": True}


class AdherenceStats(BaseModel):
    user_id: str
    total_scheduled: int
    taken: int
    missed: int
    skipped: int
    adherence_rate: float  # 0–100


# ── Schedule CRUD ────────────────────────────────────────────────────────

@router.post("/schedules", response_model=ScheduleOut, status_code=201)
async def create_schedule(
    body: ScheduleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_current_user),
):
    schedule = MedicationSchedule(user_id=current_user.id, **body.model_dump())
    db.add(schedule)
    await db.commit()
    await db.refresh(schedule)
    return schedule


@router.get("/schedules", response_model=List[ScheduleOut])
async def list_schedules(
    active_only: bool = True,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_current_user),
):
    stmt = select(MedicationSchedule).where(MedicationSchedule.user_id == current_user.id)
    if active_only:
        stmt = stmt.where(MedicationSchedule.is_active.is_(True))
    stmt = stmt.order_by(MedicationSchedule.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()


@router.delete("/schedules/{schedule_id}", status_code=204)
async def deactivate_schedule(
    schedule_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_current_user),
):
    result = await db.execute(
        select(MedicationSchedule).where(
            MedicationSchedule.id == schedule_id,
            MedicationSchedule.user_id == current_user.id,
        )
    )
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise HTTPException(404, "Schedule not found")
    schedule.is_active = False
    await db.commit()


# ── Adherence Logs ───────────────────────────────────────────────────────

@router.post("/logs", response_model=LogOut, status_code=201)
async def log_dose(
    body: LogCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_current_user),
):
    # Verify schedule exists
    result = await db.execute(
        select(MedicationSchedule).where(
            MedicationSchedule.id == body.schedule_id,
            MedicationSchedule.user_id == current_user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(404, "Schedule not found")

    log = AdherenceLog(**body.model_dump())
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


@router.get("/logs", response_model=List[LogOut])
async def list_logs(
    schedule_id: int,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_current_user),
):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = (
        select(AdherenceLog)
        .join(MedicationSchedule, MedicationSchedule.id == AdherenceLog.schedule_id)
        .where(AdherenceLog.schedule_id == schedule_id)
        .where(MedicationSchedule.user_id == current_user.id)
        .where(AdherenceLog.logged_at >= since)
        .order_by(AdherenceLog.logged_at.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()


# ── Stats ────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=AdherenceStats)
async def get_adherence_stats(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_current_user),
):
    since = datetime.now(timezone.utc) - timedelta(days=days)

    stmt = (
        select(AdherenceLog.status, func.count())
        .join(MedicationSchedule, MedicationSchedule.id == AdherenceLog.schedule_id)
        .where(MedicationSchedule.user_id == current_user.id)
        .where(AdherenceLog.logged_at >= since)
        .group_by(AdherenceLog.status)
    )
    result = await db.execute(stmt)
    counts = {row[0]: row[1] for row in result.all()}

    taken = counts.get("taken", 0)
    missed = counts.get("missed", 0)
    skipped = counts.get("skipped", 0)
    total = taken + missed + skipped
    rate = (taken / total * 100) if total > 0 else 0.0

    return AdherenceStats(
        user_id=current_user.id,
        total_scheduled=total,
        taken=taken,
        missed=missed,
        skipped=skipped,
        adherence_rate=round(rate, 1),
    )
