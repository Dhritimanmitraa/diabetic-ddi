from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import logging

from app.database import get_db
from app.models import Drug, OffsidesEffect
from app.schemas import DrugResponse
from app.services import create_interaction_service
from app.services.cache import get_cached_drug_search, cache_drug_search

router = APIRouter(
    tags=["Drugs"]
)

logger = logging.getLogger(__name__)

@router.get("/drugs", response_model=List[DrugResponse])
async def list_drugs(
    limit: int = 50,
    offset: int = 0,
    request: Request = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all drugs with pagination.
    
    Returns drugs ordered by name for browsing.
    """
    client_ip = request.client.host if request else "unknown"
    logger.info(
        {
            "event": "drug_browse",
            "path": "/drugs",
            "limit": limit,
            "offset": offset,
            "client_ip": client_ip,
        }
    )
    result = await db.execute(
        select(Drug)
        .order_by(Drug.name)
        .offset(offset)
        .limit(min(limit, 100))  # Max 100 per request
    )
    drugs = result.scalars().all()
    return [DrugResponse.model_validate(d) for d in drugs]


@router.get("/drugs/search", response_model=List[DrugResponse])
async def search_drugs(
    query: str,
    limit: int = 10,
    request: Request = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for drugs by name.
    
    Supports partial matching on drug name, generic name, and brand names.
    Results are cached for 1 hour to improve performance.
    """
    client_ip = request.client.host if request else "unknown"
    logger.info(
        {
            "event": "drug_search",
            "path": "/drugs/search",
            "query": query,
            "limit": limit,
            "client_ip": client_ip,
        }
    )
    
    # Normalize query for cache key
    normalized_query = query.strip().lower()
    effective_limit = min(limit, 50)  # Cap at 50 for caching
    
    # Try to get from cache first
    cached_results = await get_cached_drug_search(normalized_query, effective_limit)
    if cached_results is not None:
        logger.debug(f"Cache HIT for drug search: {normalized_query}")
        return [DrugResponse.model_validate(d) for d in cached_results]
    
    # Cache miss - query database
    logger.debug(f"Cache MISS for drug search: {normalized_query}")
    service = create_interaction_service(db)
    drugs = await service.search_drugs(query, effective_limit)
    
    # Convert to dict for caching (Pydantic models aren't directly JSON serializable)
    drugs_data = [
        {
            "id": d.id,
            "drugbank_id": d.drugbank_id,
            "name": d.name,
            "generic_name": d.generic_name,
            "brand_names": d.brand_names,
            "description": d.description,
            "drug_class": d.drug_class,
            "mechanism": d.mechanism,
            "indication": d.indication,
            "is_approved": d.is_approved,
            "molecular_weight": d.molecular_weight,
        }
        for d in drugs
    ]
    
    # Cache the results
    await cache_drug_search(normalized_query, effective_limit, drugs_data)
    
    return [DrugResponse.model_validate(d) for d in drugs_data]


@router.get("/drugs/{drug_id}", response_model=DrugResponse)
async def get_drug(drug_id: int, db: AsyncSession = Depends(get_db)):
    """Get drug details by ID."""
    result = await db.execute(select(Drug).where(Drug.id == drug_id))
    drug = result.scalar_one_or_none()
    
    if not drug:
        raise HTTPException(status_code=404, detail="Drug not found")
    
    return DrugResponse.model_validate(drug)


@router.get("/drugs/name/{drug_name}", response_model=DrugResponse)
async def get_drug_by_name(drug_name: str, db: AsyncSession = Depends(get_db)):
    """Get drug details by name."""
    service = create_interaction_service(db)
    drug = await service.get_drug_by_name(drug_name)
    
    if not drug:
        raise HTTPException(status_code=404, detail="Drug not found")
    
    return DrugResponse.model_validate(drug)


@router.get("/drugs/{drug_name}/side-effects")
async def get_drug_side_effects(
    drug_name: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    Get known side effects for a drug from the OffSIDES database.
    
    Returns adverse reactions and their severity levels for the specified drug.
    Useful for understanding potential risks before taking a medication.
    """
    # Normalize drug name for matching
    normalized_name = drug_name.strip().lower()
    
    # Search for side effects matching the drug name
    result = await db.execute(
        select(OffsidesEffect)
        .where(func.lower(OffsidesEffect.drug_name).contains(normalized_name))
        .limit(limit)
    )
    effects = result.scalars().all()
    
    # If no exact match, try fuzzy matching with escaped wildcards
    if not effects:
        # Escape SQL wildcard characters to prevent injection
        escaped_name = normalized_name.replace("%", "\\%").replace("_", "\\_")
        result = await db.execute(
            select(OffsidesEffect)
            .where(func.lower(OffsidesEffect.drug_name).like(f"%{escaped_name}%"))
            .limit(limit)
        )
        effects = result.scalars().all()
    
    # Group effects by severity
    effects_by_severity = {
        "severe": [],
        "moderate": [],
        "mild": [],
        "unknown": []
    }
    
    for effect in effects:
        severity_key = (effect.severity or "unknown").lower()
        if severity_key in ["severe", "major", "serious", "fatal"]:
            effects_by_severity["severe"].append(effect.effect)
        elif severity_key in ["moderate", "medium"]:
            effects_by_severity["moderate"].append(effect.effect)
        elif severity_key in ["mild", "minor"]:
            effects_by_severity["mild"].append(effect.effect)
        else:
            effects_by_severity["unknown"].append(effect.effect)
    
    # Deduplicate effects within each category
    for key in effects_by_severity:
        effects_by_severity[key] = list(set(effects_by_severity[key]))
    
    return {
        "drug_name": drug_name,
        "total_effects": len(effects),
        "effects_by_severity": effects_by_severity,
        "effects": [
            {
                "effect": e.effect,
                "severity": e.severity,
                "source": e.source
            }
            for e in effects
        ]
    }
