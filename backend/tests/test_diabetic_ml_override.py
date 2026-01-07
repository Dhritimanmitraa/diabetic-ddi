# pyright: reportMissingImports=false
import uuid

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.database import init_db


@pytest.mark.anyio
async def test_rule_override_on_low_egfr_metformin():
    await init_db()

    patient_id = f"test_lowegfr_{uuid.uuid4().hex[:6]}"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # create patient with low egfr
        resp = await client.post(
            "/diabetic/patients",
            json={
                "patient_id": patient_id,
                "age": 70,
                "gender": "F",
                "diabetes_type": "type_2",
                "labs": {"egfr": 20, "creatinine": 2.5, "potassium": 4.5},
            },
        )
        assert resp.status_code in (200, 201), resp.text

        # risk check metformin -> should rule_override to contraindicated
        resp = await client.post(
            "/diabetic/risk-check",
            json={"patient_id": patient_id, "drug_name": "metformin"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["risk_level"] == "contraindicated"
        assert data["ml_decision_source"] in ("rule_override", "rules_only")

        # If ML model is loaded and predicted safe, check that override happened
        if data["ml_decision_source"] == "rule_override":
            assert data["ml_risk_level"] == "safe"
        # If ML model is not loaded, ml_risk_level should be None
        elif data["ml_decision_source"] == "rules_only":
            assert data.get("ml_risk_level") is None
