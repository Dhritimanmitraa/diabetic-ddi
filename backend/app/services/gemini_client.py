"""Shared Gemini client wrapper with support for the new and legacy SDKs."""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Normalized Gemini response payload."""

    text: str
    model: str
    sdk: str


class GeminiClient:
    """Compatibility wrapper for Gemini text and multimodal generation."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self._client = None
        self._sdk: Optional[str] = None
        self._available = False
        self._init_client()

    def _init_client(self) -> None:
        settings = get_settings()
        api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        if not api_key:
            logger.warning("No Gemini API key configured")
            return

        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
            self._sdk = "google.genai"
            self._available = True
            logger.info("Gemini client initialized with google.genai")
            return
        except Exception as exc:
            logger.info(f"google.genai unavailable, trying legacy SDK: {exc}")

        try:
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=api_key)
            self._client = legacy_genai.GenerativeModel(self.model)
            self._sdk = "google.generativeai"
            self._available = True
            logger.info("Gemini client initialized with legacy google.generativeai")
        except Exception as exc:
            logger.warning(f"Failed to initialize Gemini client: {exc}")

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    @property
    def sdk(self) -> str:
        return self._sdk or "unavailable"

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> GeminiResponse:
        if not self.is_available:
            raise RuntimeError("Gemini client is not available")

        if self._sdk == "google.genai":
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            text = getattr(response, "text", None) or self._flatten_parts(response)
            return GeminiResponse(text=text or "", model=self.model, sdk=self._sdk)

        import google.generativeai as legacy_genai

        response = self._client.generate_content(
            prompt,
            generation_config=legacy_genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return GeminiResponse(text=(response.text or "").strip(), model=self.model, sdk=self._sdk)

    def generate_with_media(
        self,
        prompt: str,
        *,
        media_bytes: bytes,
        mime_type: str,
        temperature: float = 0.1,
        max_output_tokens: int = 2000,
    ) -> GeminiResponse:
        if not self.is_available:
            raise RuntimeError("Gemini client is not available")

        if self._sdk == "google.genai":
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=media_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            text = getattr(response, "text", None) or self._flatten_parts(response)
            return GeminiResponse(text=text or "", model=self.model, sdk=self._sdk)

        image_part = {
            "mime_type": mime_type,
            "data": base64.b64encode(media_bytes).decode("utf-8"),
        }
        response = self._client.generate_content(
            [prompt, image_part],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        return GeminiResponse(text=(response.text or "").strip(), model=self.model, sdk=self._sdk)

    def _flatten_parts(self, response: object) -> str:
        """Extract text from google.genai responses when .text is empty."""
        try:
            candidates = getattr(response, "candidates", []) or []
            parts: list[str] = []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) or []:
                    text = getattr(part, "text", None)
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        except Exception:
            return ""


_gemini_clients: dict[str, GeminiClient] = {}


def get_gemini_client(model: str = "gemini-2.0-flash") -> GeminiClient:
    """Return a cached Gemini client by model name."""
    client = _gemini_clients.get(model)
    if client is None:
        client = GeminiClient(model=model)
        _gemini_clients[model] = client
    return client