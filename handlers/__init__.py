"""Domain-specific compression handlers."""

from .legal import LegalHandler
from .logs import LogHandler
from .transcript import TranscriptHandler

__all__ = [
    "LogHandler",
    "TranscriptHandler",
    "LegalHandler",
]

