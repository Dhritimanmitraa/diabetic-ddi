"""JWT authentication routes."""
from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.services.jwt_auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    hash_password,
    require_current_user,
    rotate_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=255, description="Username or email")
    password: str = Field(..., min_length=8, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool


class TokenPairResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse


def _serialize_user(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
    )


@router.post("/register", response_model=TokenPairResponse, status_code=status.HTTP_201_CREATED)
async def register_user(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user account and return a token pair."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", body.username):
        raise HTTPException(status_code=400, detail="Username contains invalid characters")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", body.email):
        raise HTTPException(status_code=400, detail="Invalid email address")

    existing = await db.execute(
        select(User).where(or_(User.username == body.username, User.email == body.email)).limit(1)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username or email already exists")

    user = User(
        id=str(uuid.uuid4()),
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return TokenPairResponse(
        access_token=create_access_token(user),
        refresh_token=await create_refresh_token(db, user),
        user=_serialize_user(user),
    )


@router.post("/login", response_model=TokenPairResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate a user and return an access/refresh token pair."""
    user = await authenticate_user(db, body.username, body.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    return TokenPairResponse(
        access_token=create_access_token(user),
        refresh_token=await create_refresh_token(db, user),
        user=_serialize_user(user),
    )


@router.post("/refresh", response_model=TokenPairResponse)
async def refresh_access_token(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Rotate a refresh token and issue a fresh access token."""
    user, access_token, refresh_token = await rotate_refresh_token(db, body.refresh_token)
    return TokenPairResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=_serialize_user(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(require_current_user)):
    """Return the authenticated user profile."""
    return _serialize_user(current_user)
