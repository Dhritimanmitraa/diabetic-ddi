from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services import create_comparison_logger

router = APIRouter(
    tags=["History"]
)

@router.get("/history")
async def get_comparison_history(
    limit: int = 50,
    offset: int = 0,
    severity: Optional[str] = None,
    search: Optional[str] = None,
    is_safe: Optional[bool] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get drug comparison history with pagination and filtering.
    """
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)

    comparison_logger = create_comparison_logger(db)
    comparisons, total = await comparison_logger.get_comparisons(
        limit=limit,
        offset=offset,
        severity=severity,
        search=search,
        is_safe=is_safe,
    )
    
    return {
        "total": total,
        "items": [
            {
                "id": c.id,
                "drug1_name": c.drug1_name,
                "drug2_name": c.drug2_name,
                "created_at": c.timestamp,
                "has_interaction": c.has_interaction,
                "is_safe": c.is_safe,
                "severity": c.severity,
                "effect": c.effect,
            }
            for c in comparisons
        ]
    }
