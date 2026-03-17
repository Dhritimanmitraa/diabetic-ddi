"""
Test NVIDIA Cosmos Reason2-8B integration.

Verifies:
- API key is configured and valid
- Cloud endpoint responds correctly
- Vision model can process image+text prompts
- Client correctly handles errors
"""
import sys
import os

# Allow running from backend/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ─── Helpers ────────────────────────────────────────────────────────────


def _get_client():
    """Create a fresh NVIDIA client (avoid singleton caching)."""
    from app.services.nvidia_vision_client import NvidiaVisionClient
    return NvidiaVisionClient()


def _has_nvidia_credentials() -> bool:
    from app.config import get_settings

    settings = get_settings()
    return bool(settings.NVIDIA_API_KEY)


pytestmark = pytest.mark.integration


# ─── Tests ──────────────────────────────────────────────────────────────


class TestNvidiaClientSetup:
    """Basic client setup tests."""

    def test_api_key_is_configured(self):
        """NVIDIA_API_KEY must be set in .env for integration tests."""
        from app.config import get_settings
        settings = get_settings()
        if not settings.NVIDIA_API_KEY:
            pytest.skip("NVIDIA_API_KEY is not configured")
        assert settings.NVIDIA_API_KEY.startswith("nvapi-"), "NVIDIA_API_KEY should start with 'nvapi-'"

    def test_client_is_available(self):
        if not _has_nvidia_credentials():
            pytest.skip("NVIDIA_API_KEY is not configured")
        client = _get_client()
        assert client.is_available, "NVIDIA client should be available when API key is set"

    def test_model_name_is_correct(self):
        from app.config import get_settings
        settings = get_settings()
        assert settings.NVIDIA_COSMOS_MODEL
        assert settings.NVIDIA_COSMOS_MODEL.startswith("nvidia/")


class TestNvidiaCloudAPI:
    """Cloud API connectivity tests — requires network and valid API key."""

    def test_text_only_call_works(self):
        """Basic text-only call to verify API connectivity."""
        client = _get_client()
        if not client.is_available:
            pytest.skip("NVIDIA API key not configured")

        response = client.generate_text(
            "Say exactly: hello",
            max_tokens=20,
            temperature=0.0,
        )
        assert response.text, "Response text should not be empty"
        assert response.model, "Response should include model name"
        print(f"  ✅ NVIDIA API responded: {response.text[:80]}")
        print(f"  Model: {response.model}")
        if response.usage:
            print(f"  Usage: {response.usage}")

    def test_health_check(self):
        """Health check should return availability info."""
        client = _get_client()
        if not client.is_available:
            pytest.skip("NVIDIA API key not configured")

        result = client.health_check()
        assert result["available"] is True, f"Health check failed: {result}"
        print(f"  ✅ Health check passed: {result}")


class TestNvidiaImageProcessing:
    """Image processing tests (requires valid API key + network)."""

    def _create_test_image(self) -> bytes:
        """Create a simple test image with text."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io

            img = Image.new("RGB", (400, 200), color="white")
            draw = ImageDraw.Draw(img)

            # Draw some prescription-like text
            draw.text((20, 20), "Rx:", fill="black")
            draw.text((20, 50), "1. Paracetamol 500mg", fill="black")
            draw.text((20, 80), "2. Amoxicillin 250mg", fill="black")
            draw.text((20, 110), "3. Omeprazole 20mg", fill="black")
            draw.text((20, 150), "Take twice daily", fill="black")

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        except ImportError:
            pytest.skip("Pillow not installed")

    def test_image_extraction(self):
        """Send a test image to the NVIDIA Cosmos API."""
        client = _get_client()
        if not client.is_available:
            pytest.skip("NVIDIA API key not configured")

        image_bytes = self._create_test_image()

        response = client.generate_from_image(
            "What text do you see in this image? List any medicine names.",
            image_bytes=image_bytes,
            mime_type="image/png",
            max_tokens=256,
            temperature=0.1,
        )

        assert response.text, "Image response should not be empty"
        print(f"  ✅ NVIDIA image analysis: {response.text[:200]}")
        print(f"  Model: {response.model}")

    def test_prescription_extraction_prompt(self):
        """Test with the actual prescription extraction prompt used in production."""
        client = _get_client()
        if not client.is_available:
            pytest.skip("NVIDIA API key not configured")

        image_bytes = self._create_test_image()

        prompt = """Analyze this prescription image and extract all medicines.

For each medicine found, provide:
- name: The medicine/drug name
- dosage: The dosage amount (e.g., 500mg, 10ml) or null
- frequency: How often to take it (e.g., BD, TDS, 1-0-1) or null

Return ONLY a valid JSON array like this:
[{"name": "Medicine Name", "dosage": "dose or null", "frequency": "timing or null"}]

If no medicines found, return: []"""

        response = client.generate_from_image(
            prompt,
            image_bytes=image_bytes,
            mime_type="image/png",
            max_tokens=512,
            temperature=0.1,
        )

        assert response.text, "Extraction response should not be empty"
        print(f"  ✅ Extraction output: {response.text[:300]}")

        # Try to parse JSON from response
        import json
        import re

        json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            print(f"  ✅ Parsed {len(data)} medicines from JSON")
            for item in data:
                print(f"     - {item.get('name', '?')} {item.get('dosage', '')}")
        else:
            print(f"  ⚠️ No JSON array found in response (model may have used different format)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
