"""
API key auth dependency.
"""

import os
from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str = Header(None, convert_underscores=False)) -> None:
    expected = os.getenv("API_KEY")
    if expected and x_api_key == expected:
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
