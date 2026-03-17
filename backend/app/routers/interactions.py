from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.database import get_db
from app.schemas import (
    InteractionCheckRequest, InteractionCheckResponse,
    AlternativeSuggestionResponse, SeverityLevel,
    BatchInteractionRequest, BatchInteractionItem, BatchInteractionResponse,
)
from app.services import create_interaction_service, create_comparison_logger

router = APIRouter(
    tags=["Interactions"]
)

logger = logging.getLogger(__name__)

@router.post("/interactions/check", response_model=InteractionCheckResponse)
async def check_interaction(
    request: InteractionCheckRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Check if two drugs have a known interaction.
    
    Returns interaction details, safety status, and recommendations.
    All comparisons are logged for tracking.
    """
    service = create_interaction_service(db)

    # Rules-based result (for backstop and details)
    result = await service.check_interaction(request.drug1_name, request.drug2_name)

    # ML primary inference
    ml_probability = None
    ml_severity = None
    ml_predicted = None
    ml_available = False
    ml_error = None
    decision_source = "rules_only"
    try:
        from app.ml.predictor import get_predictor
        drug1 = await service.get_drug_by_name(request.drug1_name)
        drug2 = await service.get_drug_by_name(request.drug2_name)
        if drug1 and drug2:
            drug1_dict = {
                'name': drug1.name,
                'generic_name': drug1.generic_name,
                'drug_class': drug1.drug_class,
                'description': drug1.description,
                'mechanism': drug1.mechanism,
                'indication': drug1.indication,
                'molecular_weight': drug1.molecular_weight,
                'is_approved': drug1.is_approved,
            }
            drug2_dict = {
                'name': drug2.name,
                'generic_name': drug2.generic_name,
                'drug_class': drug2.drug_class,
                'description': drug2.description,
                'mechanism': drug2.mechanism,
                'indication': drug2.indication,
                'molecular_weight': drug2.molecular_weight,
                'is_approved': drug2.is_approved,
            }
            predictor = get_predictor("./models")
            if predictor.is_loaded:
                ml_available = True
                ml_res = predictor.predict(drug1_dict, drug2_dict)
                ml_probability = ml_res.interaction_probability
                ml_severity = ml_res.severity_prediction
                ml_predicted = ml_res.predicted_interaction
                decision_source = "ml_primary"
            else:
                ml_error = "ML models not loaded"
    except Exception as e:
        logger.error(f"ML prediction failed in /interactions/check: {e}")
        ml_error = str(e)

    # Hybrid gate: rules override for high-risk constraints
    rule_override_reason = None
    if result.interaction and result.interaction.severity in [SeverityLevel.CONTRAINDICATED, SeverityLevel.MAJOR]:
        final_has = True
        final_safe = False
        decision_source = "rule_override"
        rule_override_reason = f"Rule flagged {result.interaction.severity} severity interaction"
    elif ml_predicted is not None:
        final_has = bool(ml_predicted)
        final_safe = not final_has
    else:
        final_has = result.has_interaction
        final_safe = result.is_safe

    # Attach ML fields
    result.has_interaction = final_has
    result.is_safe = final_safe
    result.ml_probability = ml_probability
    result.ml_severity = ml_severity
    result.ml_decision_source = decision_source
    result.ml_available = ml_available
    result.ml_error = ml_error
    
    # Log the comparison with ML audit fields
    comparison_logger = create_comparison_logger(db)
    await comparison_logger.log_comparison(
        drug1_name=request.drug1_name,
        drug2_name=request.drug2_name,
        drug1_id=result.drug1.id if result.drug1.id != 0 else None,
        drug2_id=result.drug2.id if result.drug2.id != 0 else None,
        has_interaction=result.has_interaction,
        is_safe=result.is_safe,
        severity=result.interaction.severity if result.interaction else None,
        effect=result.interaction.effect if result.interaction else None,
        safety_message=result.safety_message,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("user-agent"),
        # ML audit fields
        ml_probability=ml_probability,
        ml_severity=ml_severity,
        ml_decision_source=decision_source,
        rule_override_reason=rule_override_reason
    )
    
    return result


@router.get("/interactions/check/{drug1}/{drug2}", response_model=InteractionCheckResponse)
async def check_interaction_get(
    drug1: str,
    drug2: str,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Check interaction between two drugs (GET endpoint).
    
    URL-encoded drug names. Internally calls the POST handler for consistent logic.
    """
    # Reuse POST handler logic for consistency (ML prediction, logging, etc.)
    request = InteractionCheckRequest(drug1_name=drug1, drug2_name=drug2)
    return await check_interaction(request, req, db)


@router.get("/interactions/drug/{drug_name}")
async def get_drug_interactions(
    drug_name: str,
    severity: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all known interactions for a specific drug.
    
    Optionally filter by severity level.
    """
    service = create_interaction_service(db)
    interactions = await service.get_all_interactions_for_drug(drug_name, severity)
    
    return {
        "drug": drug_name,
        "total_interactions": len(interactions),
        "interactions": [
            {
                "id": i.id,
                "other_drug": i.drug2.name if i.drug1.name.upper() == drug_name.upper() else i.drug1.name,
                "severity": i.severity,
                "effect": i.effect,
                "management": i.management
            }
            for i in interactions
        ]
    }


@router.post("/alternatives", response_model=AlternativeSuggestionResponse, tags=["Alternatives"])
async def get_alternatives(
    request: InteractionCheckRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get safe alternative medications when an interaction is detected.
    
    Finds similar drugs that don't interact with the other medication.
    """
    service = create_interaction_service(db)
    
    try:
        return await service.find_alternatives(
            request.drug1_name,
            request.drug2_name
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/alternatives/{drug1}/{drug2}", response_model=AlternativeSuggestionResponse, tags=["Alternatives"])
async def get_alternatives_get(
    drug1: str,
    drug2: str,
    db: AsyncSession = Depends(get_db)
):
    """Get safe alternatives (GET endpoint)."""
    service = create_interaction_service(db)
    
    try:
        return await service.find_alternatives(drug1, drug2)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/interactions/check-batch", response_model=BatchInteractionResponse)
async def check_batch_interactions(
    request: BatchInteractionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Check all pairwise interactions for a list of medications.

    Given N drugs, checks every unique pair (N*(N-1)/2 total).
    Useful for validating an entire prescription at once.
    """
    service = create_interaction_service(db)
    unique_names = list(dict.fromkeys(request.drug_names))  # dedupe, preserve order

    results: List[BatchInteractionItem] = []
    interactions_found = 0

    for d1, d2, check in await service.check_batch_interactions(unique_names):
        try:
            item = BatchInteractionItem(
                drug1_name=d1,
                drug2_name=d2,
                has_interaction=check.has_interaction,
                is_safe=check.is_safe,
                severity=check.interaction.severity if check.interaction else None,
                description=check.interaction.description if check.interaction else None,
                safety_message=check.safety_message,
            )
        except Exception as e:
            logger.warning(f"Batch check failed for {d1}-{d2}: {e}")
            item = BatchInteractionItem(
                drug1_name=d1,
                drug2_name=d2,
                has_interaction=False,
                is_safe=True,
                safety_message=f"Could not verify: {e}",
            )
        results.append(item)
        if item.has_interaction:
            interactions_found += 1

    return BatchInteractionResponse(
        drugs_checked=unique_names,
        total_pairs=len(results),
        interactions_found=interactions_found,
        results=results,
    )
