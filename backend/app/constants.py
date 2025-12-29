"""
Application-wide constants.
"""
from enum import Enum


class SeverityLevel(str, Enum):
    """Drug interaction severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"
    FATAL = "fatal"


class RiskLevel(str, Enum):
    """Drug risk levels for diabetic patients."""
    SAFE = "safe"
    CAUTION = "caution"
    HIGH_RISK = "high_risk"
    CONTRAINDICATED = "contraindicated"
    FATAL = "fatal"


# API Response Messages
class Messages:
    """Standard API response messages."""
    DRUG_NOT_FOUND = "Drug not found"
    INTERACTION_NOT_FOUND = "No interaction found"
    INVALID_INPUT = "Invalid input provided"
    SERVER_ERROR = "An internal server error occurred"
    RATE_LIMIT_EXCEEDED = "Rate limit exceeded. Please try again later."
    UNAUTHORIZED = "Invalid or missing API key"
    ML_MODEL_NOT_LOADED = "ML models not loaded. Please train models first."


# Default Limits
class Limits:
    """Default limits for queries and operations."""
    DRUG_SEARCH_LIMIT = 10
    DRUG_LIST_LIMIT = 50
    MAX_DRUG_LIST_LIMIT = 100
    HISTORY_LIMIT = 50
    MAX_HISTORY_LIMIT = 100
    OCR_FUZZY_MATCH_LIMIT = 1000  # Reduced from 10000 for performance
    MAX_OCR_TEXT_LENGTH = 1000


# Cache TTL (Time To Live) in seconds
class CacheTTL:
    """Cache expiration times."""
    DRUG_SEARCH = 300  # 5 minutes
    DRUG_DETAILS = 600  # 10 minutes
    INTERACTION_CHECK = 1800  # 30 minutes
    STATS = 60  # 1 minute
    OPENFDA_DATA = 21600  # 6 hours


# Error Codes
class ErrorCodes:
    """Standard error codes."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    UNAUTHORIZED_ERROR = "UNAUTHORIZED_ERROR"
    ML_MODEL_ERROR = "ML_MODEL_ERROR"







