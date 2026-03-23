from __future__ import annotations

import base64
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.diabetic import router as diabetic_router
from app.main import app
from app.models import ComparisonLog, Drug, DrugInteraction, OffsidesEffect, TwosidesInteraction
from app.prescription import router as prescription_router
from app.routers import admin as admin_router
from app.routers import ml_router


_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
_engine = create_async_engine(_TEST_DB_URL, echo=False)
_SessionLocal = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def _override_get_db():
    async with _SessionLocal() as session:
        yield session


@pytest_asyncio.fixture(autouse=True)
async def setup_db(monkeypatch):
    async def _noop_rate_limit(*args, **kwargs):
        return None

    monkeypatch.setattr("app.services.http_middleware.rate_limit", _noop_rate_limit)
    app.dependency_overrides[get_db] = _override_get_db
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    app.dependency_overrides.clear()
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def auth_headers(client):
    async def _create_headers(
        username: str = "user1",
        email: str = "user1@example.com",
        password: str = "Passw0rd!",
    ):
        response = await client.post(
            "/auth/register",
            json={"username": username, "email": email, "password": password},
        )
        assert response.status_code == 201
        access_token = response.json()["access_token"]
        return {"Authorization": f"Bearer {access_token}"}

    return _create_headers


@pytest_asyncio.fixture
async def seed_core_data():
    async with _SessionLocal() as session:
        session.add_all(
            [
                Drug(
                    id=1,
                    drugbank_id="DB00945",
                    name="Aspirin",
                    generic_name="Acetylsalicylic Acid",
                    brand_names='["Ecosprin"]',
                    drug_class="NSAID",
                    description="Pain relief",
                    mechanism="COX inhibitor",
                    indication="Pain",
                    is_approved=True,
                    molecular_weight=180.16,
                ),
                Drug(
                    id=2,
                    drugbank_id="DB00682",
                    name="Warfarin",
                    generic_name="Coumadin",
                    drug_class="Anticoagulant",
                    description="Anticoagulant",
                    mechanism="Vitamin K antagonist",
                    indication="Clot prevention",
                    is_approved=True,
                    molecular_weight=308.33,
                ),
                Drug(
                    id=3,
                    drugbank_id="DB00331",
                    name="Metformin",
                    generic_name="Glucophage",
                    drug_class="Biguanide",
                    description="Diabetes treatment",
                    mechanism="Lower glucose",
                    indication="Type 2 diabetes",
                    is_approved=True,
                    molecular_weight=129.16,
                ),
            ]
        )
        await session.flush()
        session.add(
            DrugInteraction(
                drug1_id=1,
                drug2_id=2,
                severity="major",
                description="Bleeding risk",
                effect="Increased bleeding",
                management="Avoid together",
            )
        )
        session.add(OffsidesEffect(drug_name="Aspirin", effect="Bleeding", source="offsides", severity="major"))
        session.add(TwosidesInteraction(drug1_name="Aspirin", drug2_name="Warfarin", effect="Bleeding", severity="major"))
        session.add(
            ComparisonLog(
                drug1_name="Aspirin",
                drug2_name="Warfarin",
                has_interaction=True,
                is_safe=False,
                severity="major",
                effect="Bleeding",
                safety_message="Unsafe",
                timestamp=datetime.now(timezone.utc),
            )
        )
        await session.commit()


def _diabetic_patient_response(patient_id: str = "PAT-1") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": 1,
        "patient_id": patient_id,
        "name": "John Doe",
        "age": 55,
        "gender": "M",
        "weight_kg": 75.0,
        "height_cm": 170.0,
        "bmi": 26.0,
        "diabetes_type": "type_2",
        "years_with_diabetes": 5,
        "hba1c": 7.1,
        "fasting_glucose": 120.0,
        "postprandial_glucose": None,
        "mean_blood_glucose": None,
        "egfr": 72.0,
        "kidney_stage": "stage_2",
        "creatinine": 1.1,
        "potassium": 4.2,
        "alt": 20.0,
        "ast": 18.0,
        "total_cholesterol": None,
        "triglycerides": None,
        "hdl_cholesterol": None,
        "ldl_cholesterol": None,
        "vldl_cholesterol": None,
        "has_nephropathy": False,
        "has_retinopathy": False,
        "has_neuropathy": False,
        "has_cardiovascular": False,
        "has_hypertension": False,
        "has_hyperlipidemia": False,
        "has_obesity": False,
        "allergies": [],
        "comorbidities": [],
        "created_at": now,
        "updated_at": now,
    }


def _drug_risk_response(drug_name: str = "Metformin", patient_id: str = "PAT-1") -> dict:
    return {
        "drug_name": drug_name,
        "risk_level": "safe",
        "risk_score": 10.0,
        "severity": "low",
        "risk_factors": [],
        "rule_references": [],
        "evidence_sources": [],
        "patient_factors": [patient_id],
        "recommendation": "Continue",
        "alternatives": [],
        "monitoring": [],
        "interactions": [],
        "validation_error": None,
        "is_safe": True,
        "is_fatal": False,
        "requires_monitoring": False,
        "ml_risk_level": None,
        "ml_probability": None,
        "ml_decision_source": None,
        "ml_model_version": None,
        "shap_explanation": None,
        "llm_explanation": None,
        "llm_analysis": None,
    }


class _StubDiabeticService:
    def __init__(self):
        self.rules = SimpleNamespace(assess_drug_risk=lambda drug, ctx, meds: SimpleNamespace(drug_name=drug))
        self.llm_checker = SimpleNamespace(check_drug_risk=self._llm_check)

    async def _llm_check(self, drug_name, patient_context, current_meds):
        return SimpleNamespace(
            risk_level="safe",
            risk_score=5.0,
            reasoning=f"{drug_name} is safe",
            key_concerns=[],
            monitoring_needed=[],
            model_used="stub-llm",
            was_fallback=False,
        )

    def _patient_to_response(self, patient):
        return _diabetic_patient_response(patient.patient_id)

    def _build_patient_context(self, patient):
        return {"patient_id": patient.patient_id, "egfr": patient.egfr}

    def _assessment_to_response(self, assessment):
        return _drug_risk_response(getattr(assessment, "drug_name", "Metformin"))

    async def create_patient(self, data, user_id=None):
        return SimpleNamespace(patient_id=data.patient_id)

    async def list_patients(self, limit, offset, user_id=None):
        return [SimpleNamespace(patient_id="PAT-1")], 1

    async def get_patient(self, patient_id, user_id=None):
        if patient_id == "missing":
            return None
        return SimpleNamespace(
            patient_id=patient_id,
            name="John Doe",
            age=55,
            gender="M",
            diabetes_type="type_2",
            egfr=72.0,
            hba1c=7.1,
            fasting_glucose=120.0,
            potassium=4.2,
            creatinine=1.1,
            has_nephropathy=False,
            has_retinopathy=False,
            has_neuropathy=False,
            has_cardiovascular=False,
        )

    async def update_patient(self, patient_id, data, user_id=None):
        return None if patient_id == "missing" else SimpleNamespace(patient_id=patient_id)

    async def delete_patient(self, patient_id, user_id=None):
        return patient_id != "missing"

    async def add_medication(self, patient_id, data, user_id=None):
        if patient_id == "missing":
            return None
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            id=1,
            drug_name=data.drug_name,
            generic_name=data.generic_name,
            drug_class=data.drug_class,
            dose=data.dose,
            dosage=data.dose,
            frequency=data.frequency,
            route=data.route,
            indication=data.indication,
            is_diabetes_medication=data.is_diabetes_medication,
            is_active=True,
            start_date=now,
        )

    async def get_patient_medications(self, patient_id, active_only=True, user_id=None):
        if patient_id == "empty":
            return []
        now = datetime.now(timezone.utc)
        return [
            SimpleNamespace(
                id=1,
                drug_name="Metformin",
                generic_name="Metformin",
                drug_class="Biguanide",
                dose="500mg",
                dosage="500mg",
                frequency="BID",
                route="oral",
                indication="Diabetes",
                is_diabetes_medication=True,
                is_active=True,
                start_date=now,
            )
        ]

    async def remove_medication(self, patient_id, medication_id, user_id=None):
        return patient_id != "missing"

    async def check_drug_risk(self, patient_id, drug_name, user_id=None):
        return None if patient_id == "missing" else _drug_risk_response(drug_name, patient_id)

    async def check_all_medications(self, patient_id, medications, user_id=None):
        if patient_id == "missing":
            return None
        return {
            "patient_id": patient_id,
            "total_medications": 1,
            "safe_count": 1,
            "caution_count": 0,
            "high_risk_count": 0,
            "contraindicated_count": 0,
            "fatal_count": 0,
            "assessments": [_drug_risk_response("Metformin", patient_id)],
            "overall_risk_level": "safe",
            "critical_alerts": [],
            "recommendations": ["Continue"],
        }

    async def find_safe_alternatives(self, patient_id, drug_name, user_id=None):
        if patient_id == "missing":
            return None
        return {
            "original_drug": drug_name,
            "original_risk_level": "high_risk",
            "alternatives": [{"drug": "Metformin XR", "risk_level": "safe", "risk_score": 5.0, "considerations": []}],
        }

    async def generate_patient_report(self, patient_id, include_alternatives=True, user_id=None):
        if patient_id == "missing":
            return None
        return SimpleNamespace(
            patient=_diabetic_patient_response(patient_id),
            report_generated_at=datetime.now(timezone.utc).isoformat(),
            current_medications=[
                {
                    "id": 1,
                    "drug_name": "Metformin",
                    "generic_name": "Metformin",
                    "drug_class": "Biguanide",
                    "dose": "500mg",
                    "frequency": "BID",
                    "route": "oral",
                    "indication": "Diabetes",
                    "is_diabetes_medication": True,
                    "is_active": True,
                    "start_date": datetime.now(timezone.utc).isoformat(),
                }
            ],
            medication_assessments=[
                SimpleNamespace(
                    **_drug_risk_response("Metformin", patient_id),
                    recommendations=["Continue"],
                )
            ],
            fatal_risks=[],
            contraindicated_drugs=[],
            high_risk_drugs=[],
            recommended_alternatives={},
            monitoring_plan=[],
            overall_safety_score=95.0,
            action_required=False,
            summary="Stable",
        )


class _StubPrescriptionService:
    async def upload_and_process(self, **kwargs):
        return {
            "id": 1,
            "status": "completed",
            "message": "ok",
            "medicines": [],
            "raw_text": "take once daily",
            "extraction_confidence": 0.9,
            "vision_model_used": "stub",
        }

    async def list_prescriptions(self, user_id, limit=20, offset=0):
        return ([], 0)

    async def get_prescription(self, prescription_id, user_id):
        if prescription_id == 404:
            return None
        return {
            "id": prescription_id,
            "filename": "rx.jpg",
            "file_type": "image/jpeg",
            "status": "completed",
            "extraction_confidence": 0.9,
            "vision_model_used": "stub",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "medicines": [],
            "error_message": None,
        }

    async def delete_prescription(self, prescription_id, user_id):
        return prescription_id != 404

    async def chat(self, prescription_id, message, user_id):
        if prescription_id == 404:
            return None
        return {
            "prescription_id": prescription_id,
            "user_message": message,
            "assistant_message": "Take after food",
            "model_used": "stub-rag",
            "retrieved_context": "context",
        }

    async def get_chat_history(self, prescription_id, user_id):
        if prescription_id == 404:
            return None
        return {
            "prescription_id": prescription_id,
            "messages": [
                {"role": "user", "content": "When?"},
                {"role": "assistant", "content": "After food"},
            ],
        }


@pytest_asyncio.fixture
async def diabetic_service_override():
    stub = _StubDiabeticService()

    async def _override():
        return stub

    app.dependency_overrides[diabetic_router.get_service] = _override
    yield stub
    app.dependency_overrides.pop(diabetic_router.get_service, None)


@pytest_asyncio.fixture
async def prescription_service_override():
    stub = _StubPrescriptionService()

    def _override():
        return stub

    app.dependency_overrides[prescription_router.get_service] = _override
    yield stub
    app.dependency_overrides.pop(prescription_router.get_service, None)


class TestHistoryEndpoints:
    @pytest.mark.asyncio
    async def test_history_endpoint(self, client, seed_core_data):
        response = await client.get("/history")
        assert response.status_code == 200
        assert response.json()["total"] >= 1


class TestAdherenceEndpoints:
    @pytest.mark.asyncio
    async def test_adherence_crud_and_stats(self, client, auth_headers):
        headers = await auth_headers(username="adherence_user", email="adherence@example.com")

        create_resp = await client.post(
            "/adherence/schedules",
            json={"drug_name": "Metformin", "dosage": "500mg"},
            headers=headers,
        )
        assert create_resp.status_code == 201
        schedule_id = create_resp.json()["id"]

        list_resp = await client.get("/adherence/schedules", headers=headers)
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 1

        log_resp = await client.post(
            "/adherence/logs",
            json={"schedule_id": schedule_id, "status": "taken"},
            headers=headers,
        )
        assert log_resp.status_code == 201

        logs_resp = await client.get("/adherence/logs", params={"schedule_id": schedule_id}, headers=headers)
        assert logs_resp.status_code == 200
        assert len(logs_resp.json()) == 1

        stats_resp = await client.get("/adherence/stats", headers=headers)
        assert stats_resp.status_code == 200
        assert stats_resp.json()["taken"] == 1

        deactivate_resp = await client.delete(f"/adherence/schedules/{schedule_id}", headers=headers)
        assert deactivate_resp.status_code == 204


class TestOcrEndpoints:
    @pytest.mark.asyncio
    async def test_ocr_extract_and_upload(self, client, seed_core_data, monkeypatch):
        class StubOCR:
            def extract_from_base64(self, image_base64):
                return "Aspirin", ["Aspirin"], 0.88

            def find_similar_drug_names(self, *args, **kwargs):
                return [("Aspirin", 0.9)]

        monkeypatch.setattr("app.routers.ocr.create_ocr_service", lambda *_: StubOCR())

        response = await client.post("/ocr/extract", json={"image_base64": base64.b64encode(b"img").decode()})
        assert response.status_code == 200
        assert response.json()["detected_drugs"][0] == "Aspirin"

        upload_resp = await client.post("/ocr/upload", files={"file": ("pill.jpg", b"binary", "image/jpeg")})
        assert upload_resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ocr_upload_rejects_non_image(self, client):
        response = await client.post("/ocr/upload", files={"file": ("x.txt", b"bad", "text/plain")})
        assert response.status_code == 400


class TestMlEndpoints:
    @pytest.mark.asyncio
    async def test_ml_status_info_predict_and_comparison(self, client, seed_core_data, monkeypatch):
        class StubPredictResult:
            def to_dict(self):
                return {"interaction_probability": 0.8, "predicted_interaction": True}

        class StubPredictor:
            is_loaded = True

            def get_model_info(self):
                return {
                    "models_loaded": ["rf"],
                    "optimal_threshold": 0.42,
                    "threshold_method": "youden",
                    "model_metrics": {"rf": {"auc": 0.9}},
                }

            def get_feature_importance(self):
                return [{"feature": "class_overlap", "importance": 0.7}]

            def predict(self, drug1, drug2):
                return StubPredictResult()

        monkeypatch.setattr("app.ml.predictor.get_predictor", lambda *_: StubPredictor())
        monkeypatch.setattr("app.ml.scheduler._scheduler", None, raising=False)
        monkeypatch.setattr("app.ml.scheduler.RETRAIN_CHECK_HOURS", 24, raising=False)
        monkeypatch.setattr("app.ml.scheduler.RETRAIN_MIN_NEW_ROWS", 100, raising=False)

        status_resp = await client.get("/ml/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["ml_available"] is True

        info_resp = await client.get("/ml/model-info")
        assert info_resp.status_code == 200
        assert info_resp.json()["status"] == "loaded"

        predict_resp = await client.post("/ml/predict", json={"drug1_name": "Aspirin", "drug2_name": "Warfarin"})
        assert predict_resp.status_code == 200
        assert predict_resp.json()["predicted_interaction"] is True

        comparison_resp = await client.get("/ml/comparison")
        assert comparison_resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ml_train_and_admin_jobs(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "test-key")
        from app.config import get_settings

        get_settings.cache_clear()
        async def _start_job(*args, **kwargs):
            return "job-1"

        async def _list_jobs(*args, **kwargs):
            return [{"id": "job-1"}]

        async def _has_active_job(*args, **kwargs):
            return False

        monkeypatch.setattr(ml_router, "has_active_job", _has_active_job)
        monkeypatch.setattr(ml_router, "start_tracked_job", _start_job)
        monkeypatch.setattr(ml_router, "_run_training", lambda *args, **kwargs: None)
        monkeypatch.setattr(admin_router, "get_all_jobs", _list_jobs)

        train_resp = await client.post(
            "/ml/train",
            json={"n_trials": 5, "run_comparison": False},
            headers={"X-API-Key": "test-key"},
        )
        assert train_resp.status_code == 200
        assert "job-1" in train_resp.json()["message"]

        jobs_resp = await client.get("/admin/jobs", headers={"X-API-Key": "test-key"})
        assert jobs_resp.status_code == 200
        assert jobs_resp.json()["jobs"][0]["id"] == "job-1"
        get_settings.cache_clear()


class TestAuthAndMetricsEndpoints:
    @pytest.mark.asyncio
    async def test_register_login_refresh_me_and_metrics(self, client):
        register_resp = await client.post(
            "/auth/register",
            json={
                "username": "architect",
                "email": "architect@example.com",
                "password": "Passw0rd!",
            },
        )
        assert register_resp.status_code == 201
        token_pair = register_resp.json()
        assert token_pair["token_type"] == "bearer"

        me_resp = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token_pair['access_token']}"},
        )
        assert me_resp.status_code == 200
        assert me_resp.json()["username"] == "architect"

        login_resp = await client.post(
            "/auth/login",
            json={"username": "architect", "password": "Passw0rd!"},
        )
        assert login_resp.status_code == 200
        assert "X-Process-Time-MS" in login_resp.headers
        assert "X-Latency-Target-MS" in login_resp.headers

        refresh_resp = await client.post(
            "/auth/refresh",
            json={"refresh_token": login_resp.json()["refresh_token"]},
        )
        assert refresh_resp.status_code == 200
        assert refresh_resp.json()["access_token"]

        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "http_requests_total" in metrics_resp.text
        assert "http_request_duration_seconds" in metrics_resp.text
        assert "http_slow_requests_total" in metrics_resp.text


class TestDiabeticEndpoints:
    @pytest.mark.asyncio
    async def test_patient_medication_and_risk_endpoints(self, client, diabetic_service_override, auth_headers):
        headers = await auth_headers(username="diabetic_user", email="diabetic@example.com")

        create_resp = await client.post("/diabetic/patients", json={"patient_id": "PAT-1"}, headers=headers)
        assert create_resp.status_code == 201

        list_resp = await client.get("/diabetic/patients", headers=headers)
        assert list_resp.status_code == 200

        get_resp = await client.get("/diabetic/patients/PAT-1", headers=headers)
        assert get_resp.status_code == 200

        patch_resp = await client.patch("/diabetic/patients/PAT-1", json={"age": 60}, headers=headers)
        assert patch_resp.status_code == 200

        med_resp = await client.post(
            "/diabetic/patients/PAT-1/medications",
            json={"drug_name": "Metformin", "dose": "500mg"},
            headers=headers,
        )
        assert med_resp.status_code == 201

        meds_resp = await client.get("/diabetic/patients/PAT-1/medications", headers=headers)
        assert meds_resp.status_code == 200

        risk_resp = await client.post(
            "/diabetic/risk-check",
            json={"patient_id": "PAT-1", "drug_name": "Metformin"},
            headers=headers,
        )
        assert risk_resp.status_code == 200

        llm_resp = await client.post(
            "/diabetic/risk-check/llm",
            json={"patient_id": "PAT-1", "drug_name": "Metformin"},
            headers=headers,
        )
        assert llm_resp.status_code == 200
        assert llm_resp.json()["llm_analysis"]["model_used"] == "stub-llm"

        med_list_resp = await client.post(
            "/diabetic/medication-list-check",
            json={"patient_id": "PAT-1"},
            headers=headers,
        )
        assert med_list_resp.status_code == 200

        alt_resp = await client.post(
            "/diabetic/alternatives",
            json={"patient_id": "PAT-1", "drug_name": "Insulin"},
            headers=headers,
        )
        assert alt_resp.status_code == 200

        report_post = await client.post("/diabetic/report", json={"patient_id": "PAT-1"}, headers=headers)
        assert report_post.status_code == 200

        report_get = await client.get("/diabetic/report/PAT-1", headers=headers)
        assert report_get.status_code == 200

        remove_med = await client.delete("/diabetic/patients/PAT-1/medications/1", headers=headers)
        assert remove_med.status_code == 204

        delete_resp = await client.delete("/diabetic/patients/PAT-1", headers=headers)
        assert delete_resp.status_code == 204

    @pytest.mark.asyncio
    async def test_diabetic_pdf_analysis_and_info_endpoints(
        self,
        client,
        diabetic_service_override,
        seed_core_data,
        monkeypatch,
        auth_headers,
    ):
        headers = await auth_headers(username="diabetic_pdf_user", email="diabetic-pdf@example.com")

        monkeypatch.setattr(
            "app.services.pdf_generator.generate_patient_report_pdf",
            lambda **kwargs: b"%PDF-1.4 test",
        )

        class StubExtractedValues:
            def model_dump(self):
                return {"patient": {"name": "John Doe"}}

            patient = SimpleNamespace(name="John Doe", patient_id="PAT-1", age=50, gender="M")
            glucose = SimpleNamespace(hba1c=6.8, fasting_glucose=110.0, postprandial_glucose=None, mean_blood_glucose=None)
            kidney = SimpleNamespace(egfr=70.0, creatinine=1.0)
            potassium = 4.2
            lipid = SimpleNamespace(total_cholesterol=None, triglycerides=None, hdl_cholesterol=None, ldl_cholesterol=None, vldl_cholesterol=None)
            liver = SimpleNamespace(alt=None, ast=None)

        class StubHealthSummary:
            def model_dump(self):
                return {"summary": "stable"}

        class StubAnalyzerResult:
            success = True
            extracted_values = StubExtractedValues()
            health_summary = StubHealthSummary()
            model_used = "stub-gemini"
            error = None

        class StubAnalyzer:
            gemini_available = True

            async def analyze_report(self, *args, **kwargs):
                return StubAnalyzerResult()

            async def get_personalized_ddi_analysis(self, patient_data, drug_name):
                return {"risk": "low", "drug": drug_name}

        monkeypatch.setattr("app.diabetic.lab_report_analyzer.get_lab_report_analyzer", lambda: StubAnalyzer())

        pdf_resp = await client.get("/diabetic/report/PAT-1/pdf", headers=headers)
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers["content-type"].startswith("application/pdf")

        analyze_resp = await client.post(
            "/diabetic/analyze-report",
            files={"file": ("report.jpg", b"img", "image/jpeg")},
            headers=headers,
        )
        assert analyze_resp.status_code == 200
        assert analyze_resp.json()["success"] is True

        personalized_resp = await client.post(
            "/diabetic/analyze-report/personalized-ddi",
            params={"patient_id": "PAT-1", "drug_name": "Metformin"},
            headers=headers,
        )
        assert personalized_resp.status_code == 200

        analyzer_status = await client.get("/diabetic/analyzer-status")
        assert analyzer_status.status_code == 200

        quick_check = await client.get("/diabetic/quick-check/PAT-1/Metformin", headers=headers)
        assert quick_check.status_code == 200

        rules_info = await client.get("/diabetic/rules/info")
        assert rules_info.status_code == 200

        model_info = await client.get("/diabetic/model-info")
        assert model_info.status_code == 200

        classes_resp = await client.get("/diabetic/drug-classes")
        assert classes_resp.status_code == 200

        twosides_resp = await client.get("/diabetic/twosides/count")
        assert twosides_resp.status_code == 200

        offsides_resp = await client.get("/diabetic/offsides/count")
        assert offsides_resp.status_code == 200

        search_resp = await client.get("/diabetic/drugs/search", params={"query": "asp"})
        assert search_resp.status_code == 200

        preview_resp = await client.post(
            "/diabetic/rules/preview",
            json={"patient": {"diabetes_type": "type_2", "age": 55}, "drugs": ["Metformin"]},
            headers=headers,
        )
        assert preview_resp.status_code == 200

        food_resp = await client.get("/diabetic/food-interactions/warfarin")
        assert food_resp.status_code == 200

        categories_resp = await client.get("/diabetic/food-categories")
        assert categories_resp.status_code == 200

        category_drugs_resp = await client.get("/diabetic/food-categories/grapefruit/drugs")
        assert category_drugs_resp.status_code in {200, 404}

        patient_food_resp = await client.get("/diabetic/patients/PAT-1/food-interactions", headers=headers)
        assert patient_food_resp.status_code == 200


class TestPrescriptionEndpoints:
    @pytest.mark.asyncio
    async def test_prescription_crud_chat_and_health(
        self,
        client,
        prescription_service_override,
        monkeypatch,
        auth_headers,
    ):
        headers = await auth_headers(username="rxuser", email="rxuser@example.com")

        upload_resp = await client.post(
            "/prescription/upload",
            files={"file": ("rx.jpg", b"img", "image/jpeg")},
            headers=headers,
        )
        assert upload_resp.status_code == 200

        upload_b64_resp = await client.post(
            "/prescription/upload/base64",
            data={"image_base64": base64.b64encode(b"img").decode(), "filename": "rx.jpg"},
            headers=headers,
        )
        assert upload_b64_resp.status_code == 200

        history_resp = await client.get("/prescription/history", headers=headers)
        assert history_resp.status_code == 200

        get_resp = await client.get("/prescription/1", headers=headers)
        assert get_resp.status_code == 200

        delete_resp = await client.delete("/prescription/1", headers=headers)
        assert delete_resp.status_code == 200

        chat_resp = await client.post(
            "/prescription/chat",
            json={"prescription_id": 1, "message": "When?"},
            headers=headers,
        )
        assert chat_resp.status_code == 200

        chat_history_resp = await client.get("/prescription/1/chat-history", headers=headers)
        assert chat_history_resp.status_code == 200

        unauthorized_resp = await client.get("/prescription/history")
        assert unauthorized_resp.status_code == 401

        class StubVision:
            nvidia_available = False
            gemini_available = True

        class StubRag:
            chroma_client = object()

        class StubLlm:
            gemini_available = True

        monkeypatch.setattr("app.prescription.vision_ocr.get_vision_service", lambda: StubVision())
        monkeypatch.setattr("app.prescription.rag_service.get_rag_service", lambda: StubRag())
        monkeypatch.setattr("app.prescription.rag_service.get_llm_service", lambda: StubLlm())

        health_resp = await client.get("/prescription/health/status")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_prescription_interaction_check(self, client, seed_core_data):
        response = await client.post(
            "/prescription/check-interactions",
            json={"drug_names": ["Aspirin", "Warfarin", "Aspirin"]},
        )
        assert response.status_code == 200
        assert response.json()["total_interactions"] >= 1
