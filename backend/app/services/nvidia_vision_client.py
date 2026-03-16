"""
NVIDIA Cosmos Reason2-8B Vision-Language Model Client.

Uses the NVIDIA cloud API (OpenAI-compatible) at integrate.api.nvidia.com
for prescription image analysis and OCR.

Cosmos Reason2-8B is a vision language model that excels at understanding
the physical world using structured reasoning on images and videos.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class NvidiaVisionResponse:
    """Normalized NVIDIA vision response."""

    text: str
    model: str
    usage: Optional[dict] = None


class NvidiaVisionClient:
    """Client for NVIDIA Cosmos Reason2-8B vision-language model.

    Uses the NVIDIA cloud API (OpenAI-compatible chat/completions endpoint).
    """

    def __init__(self):
        settings = get_settings()
        self.api_key: str = settings.NVIDIA_API_KEY
        self.base_url: str = settings.NVIDIA_NIM_BASE_URL.rstrip("/")
        self.model: str = settings.NVIDIA_COSMOS_MODEL
        self._available: bool = bool(self.api_key)

        if self._available:
            logger.info(
                "NVIDIA Cosmos client ready (model=%s, endpoint=%s)",
                self.model,
                self.base_url,
            )
        else:
            logger.warning("NVIDIA API key not configured — Cosmos client disabled")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate_from_image(
        self,
        prompt: str,
        *,
        image_bytes: bytes,
        mime_type: str = "image/png",
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> NvidiaVisionResponse:
        """Send an image + text prompt to the Cosmos model.

        Args:
            prompt: Text question / instruction.
            image_bytes: Raw image bytes.
            mime_type: MIME type of the image (image/png, image/jpeg, etc.).
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            NvidiaVisionResponse with the model's text answer.
        """
        if not self._available:
            raise RuntimeError("NVIDIA client is not available (no API key)")

        # Base64-encode the image for the data-URI approach
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{b64}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.info("Calling NVIDIA Cosmos API: %s (model=%s)", url, self.model)

        response = httpx.post(
            url,
            json=payload,
            headers=headers,
            timeout=60.0,  # Vision models can be slow
        )

        if response.status_code != 200:
            error_detail = response.text[:500]
            logger.error(
                "NVIDIA API error %d: %s", response.status_code, error_detail
            )
            raise RuntimeError(
                f"NVIDIA API returned {response.status_code}: {error_detail}"
            )

        data = response.json()

        # Parse OpenAI-compatible response
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("NVIDIA API returned no choices")

        text = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage")

        return NvidiaVisionResponse(
            text=text.strip(),
            model=data.get("model", self.model),
            usage=usage,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> NvidiaVisionResponse:
        """Send a text-only prompt to the Cosmos model.

        Args:
            prompt: Text question / instruction.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            NvidiaVisionResponse with the model's text answer.
        """
        if not self._available:
            raise RuntimeError("NVIDIA client is not available (no API key)")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)

        if response.status_code != 200:
            raise RuntimeError(
                f"NVIDIA API returned {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("NVIDIA API returned no choices")

        text = choices[0].get("message", {}).get("content", "")
        return NvidiaVisionResponse(
            text=text.strip(),
            model=data.get("model", self.model),
            usage=data.get("usage"),
        )

    def health_check(self) -> dict:
        """Quick health check against the NVIDIA API.

        Returns:
            Dict with availability status and details.
        """
        if not self._available:
            return {"available": False, "reason": "No NVIDIA_API_KEY configured"}

        try:
            # Lightweight text-only call
            resp = self.generate_text(
                "Reply with exactly: OK",
                max_tokens=10,
                temperature=0.0,
            )
            return {
                "available": True,
                "model": resp.model,
                "response": resp.text[:50],
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc)[:200]}


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_nvidia_client: Optional[NvidiaVisionClient] = None


def get_nvidia_vision_client() -> NvidiaVisionClient:
    """Return a cached NVIDIA Vision client singleton."""
    global _nvidia_client
    if _nvidia_client is None:
        _nvidia_client = NvidiaVisionClient()
    return _nvidia_client
