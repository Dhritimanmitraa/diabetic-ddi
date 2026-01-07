"""
Comparison Logger Service

Tracks and logs all drug interaction comparisons made through the application.
Saves to both database and JSON file for easy access.
"""

import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, or_
import logging

from app.models import ComparisonLog, Drug

logger = logging.getLogger(__name__)

# Log file path
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
)
LOG_FILE = os.path.join(LOG_DIR, "comparison_history.json")


class ComparisonLogger:
    """Service for logging drug comparisons."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        """Ensure log directory exists."""
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                json.dump({"comparisons": [], "summary": {}}, f, indent=2)

    async def log_comparison(
        self,
        drug1_name: str,
        drug2_name: str,
        drug1_id: Optional[int],
        drug2_id: Optional[int],
        has_interaction: bool,
        is_safe: bool,
        severity: Optional[str] = None,
        effect: Optional[str] = None,
        safety_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        # ML Audit fields
        ml_probability: Optional[float] = None,
        ml_severity: Optional[str] = None,
        ml_decision_source: Optional[str] = None,
        ml_model_version: Optional[str] = None,
        rule_override_reason: Optional[str] = None,
    ) -> ComparisonLog:
        """
        Log a drug comparison to database and file.

        Args:
            drug1_name: Name of first drug
            drug2_name: Name of second drug
            drug1_id: Database ID of first drug (if found)
            drug2_id: Database ID of second drug (if found)
            has_interaction: Whether an interaction was found
            is_safe: Whether the combination is safe
            severity: Interaction severity level
            effect: Description of interaction effect
            safety_message: Safety message shown to user
            ip_address: Client IP address
            user_agent: Client user agent string
            ml_probability: ML interaction probability (0-1)
            ml_severity: ML predicted severity
            ml_decision_source: Decision source (ml_primary, rule_override, rules_only)
            ml_model_version: Version of ML model used
            rule_override_reason: Reason for rule override (if applicable)

        Returns:
            ComparisonLog object
        """
        # Create database record
        log_entry = ComparisonLog(
            drug1_name=drug1_name,
            drug2_name=drug2_name,
            drug1_id=drug1_id,
            drug2_id=drug2_id,
            has_interaction=has_interaction,
            is_safe=is_safe,
            severity=severity,
            effect=effect,
            safety_message=safety_message,
            ip_address=ip_address,
            user_agent=user_agent,
            ml_probability=ml_probability,
            ml_severity=ml_severity,
            ml_decision_source=ml_decision_source,
            ml_model_version=ml_model_version,
            rule_override_reason=rule_override_reason,
            timestamp=datetime.utcnow(),
        )

        self.db.add(log_entry)
        await self.db.commit()
        await self.db.refresh(log_entry)

        # Also save to JSON file
        await self._append_to_json_log(log_entry)

        # Log with ML decision info
        decision_info = f" [ML:{ml_decision_source}]" if ml_decision_source else ""
        logger.info(
            f"Logged comparison: {drug1_name} + {drug2_name} = {'INTERACTION' if has_interaction else 'SAFE'}{decision_info}"
        )

        return log_entry

    async def _append_to_json_log(self, log_entry: ComparisonLog):
        """Append comparison to JSON log file."""
        try:
            # Read existing data
            with open(LOG_FILE, "r") as f:
                data = json.load(f)

            # Add new entry
            entry = {
                "id": log_entry.id,
                "timestamp": log_entry.timestamp.isoformat(),
                "drug1": {"name": log_entry.drug1_name, "id": log_entry.drug1_id},
                "drug2": {"name": log_entry.drug2_name, "id": log_entry.drug2_id},
                "result": {
                    "has_interaction": log_entry.has_interaction,
                    "is_safe": log_entry.is_safe,
                    "severity": log_entry.severity,
                    "effect": log_entry.effect,
                    "safety_message": log_entry.safety_message,
                },
                "ml_decision": {
                    "probability": log_entry.ml_probability,
                    "severity": log_entry.ml_severity,
                    "decision_source": log_entry.ml_decision_source,
                    "model_version": log_entry.ml_model_version,
                    "rule_override_reason": log_entry.rule_override_reason,
                },
                "client": {
                    "ip_address": log_entry.ip_address,
                    "user_agent": log_entry.user_agent,
                },
            }

            data["comparisons"].append(entry)

            # Update summary
            data["summary"] = await self._calculate_summary()

            # Write back
            with open(LOG_FILE, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error writing to JSON log: {e}")

    async def _calculate_summary(self) -> Dict:
        """Calculate summary statistics."""
        total = await self.db.execute(select(func.count(ComparisonLog.id)))
        total_count = total.scalar()

        safe = await self.db.execute(
            select(func.count(ComparisonLog.id)).where(ComparisonLog.is_safe == True)
        )
        safe_count = safe.scalar()

        unsafe = await self.db.execute(
            select(func.count(ComparisonLog.id)).where(ComparisonLog.is_safe == False)
        )
        unsafe_count = unsafe.scalar()

        # Count by severity
        severity_counts = {}
        for sev in ["minor", "moderate", "major", "contraindicated"]:
            count = await self.db.execute(
                select(func.count(ComparisonLog.id)).where(
                    ComparisonLog.severity == sev
                )
            )
            severity_counts[sev] = count.scalar()

        return {
            "total_comparisons": total_count,
            "safe_combinations": safe_count,
            "unsafe_combinations": unsafe_count,
            "by_severity": severity_counts,
            "last_updated": datetime.utcnow().isoformat(),
        }

    async def get_comparisons(
        self,
        limit: int = 50,
        offset: int = 0,
        severity: Optional[str] = None,
        search: Optional[str] = None,
        is_safe: Optional[bool] = None,
    ) -> Tuple[List[ComparisonLog], int]:
        """Get comparisons with pagination and filters."""
        conditions = []
        if severity:
            conditions.append(ComparisonLog.severity == severity)
        if is_safe is not None:
            conditions.append(ComparisonLog.is_safe == is_safe)
        if search:
            search_upper = f"%{search.upper()}%"
            conditions.append(
                or_(
                    func.upper(ComparisonLog.drug1_name).like(search_upper),
                    func.upper(ComparisonLog.drug2_name).like(search_upper),
                )
            )

        base_query = select(ComparisonLog)
        count_query = select(func.count(ComparisonLog.id))

        if conditions:
            condition = and_(*conditions)
            base_query = base_query.where(condition)
            count_query = count_query.where(condition)

        base_query = (
            base_query.order_by(desc(ComparisonLog.timestamp))
            .offset(offset)
            .limit(limit)
        )

        rows = await self.db.execute(base_query)
        total = await self.db.execute(count_query)
        return list(rows.scalars().all()), total.scalar() or 0

    async def get_comparison_stats(self) -> Dict:
        """Get comparison statistics."""
        return await self._calculate_summary()

    async def get_most_checked_drugs(self, limit: int = 10) -> List[Dict]:
        """Get the most frequently checked drugs."""
        # This is a simplified version - counts drug1 appearances
        result = await self.db.execute(
            select(
                ComparisonLog.drug1_name, func.count(ComparisonLog.id).label("count")
            )
            .group_by(ComparisonLog.drug1_name)
            .order_by(desc("count"))
            .limit(limit)
        )

        drugs = []
        for row in result:
            drugs.append({"drug_name": row[0], "check_count": row[1]})

        return drugs

    async def get_dangerous_combinations_found(self) -> List[Dict]:
        """Get all dangerous (major/contraindicated) combinations that were checked."""
        result = await self.db.execute(
            select(ComparisonLog)
            .where(ComparisonLog.severity.in_(["major", "contraindicated"]))
            .order_by(desc(ComparisonLog.timestamp))
        )

        combinations = []
        for log in result.scalars():
            combinations.append(
                {
                    "drug1": log.drug1_name,
                    "drug2": log.drug2_name,
                    "severity": log.severity,
                    "effect": log.effect,
                    "timestamp": log.timestamp.isoformat(),
                }
            )

        return combinations


def create_comparison_logger(db: AsyncSession) -> ComparisonLogger:
    """Factory function to create comparison logger."""
    return ComparisonLogger(db)
