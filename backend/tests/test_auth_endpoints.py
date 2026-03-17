from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app


TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
engine = create_async_engine(TEST_DB_URL, echo=False)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with SessionLocal() as session:
        yield session


@pytest_asyncio.fixture(autouse=True)
async def setup_db(monkeypatch):
    async def _noop_rate_limit(*args, **kwargs):
        return None

    monkeypatch.setattr("app.services.http_middleware.rate_limit", _noop_rate_limit)
    app.dependency_overrides[get_db] = override_get_db
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_register_login_refresh_and_profile(client: AsyncClient):
    register_response = await client.post(
        "/auth/register",
        json={
            "username": "tester",
            "email": "tester@example.com",
            "password": "TestPassword123!",
        },
    )
    assert register_response.status_code == 201
    register_payload = register_response.json()
    assert register_payload["access_token"]
    assert register_payload["refresh_token"]

    me_response = await client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {register_payload['access_token']}"},
    )
    assert me_response.status_code == 200
    assert me_response.json()["username"] == "tester"

    login_response = await client.post(
        "/auth/login",
        json={"username": "tester", "password": "TestPassword123!"},
    )
    assert login_response.status_code == 200
    login_payload = login_response.json()
    assert login_payload["user"]["email"] == "tester@example.com"

    refresh_response = await client.post(
        "/auth/refresh",
        json={"refresh_token": login_payload["refresh_token"]},
    )
    assert refresh_response.status_code == 200
    refresh_payload = refresh_response.json()
    assert refresh_payload["access_token"] != ""
    assert refresh_payload["refresh_token"] != login_payload["refresh_token"]


@pytest.mark.asyncio
async def test_diabetic_routes_require_auth(client: AsyncClient):
    response = await client.get("/diabetic/patients")
    assert response.status_code == 401
