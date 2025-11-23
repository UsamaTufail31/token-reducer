"""Content type identification for intelligent compression routing."""

import re
from enum import Enum
from typing import Dict, List


class ContentType(Enum):
    """Types of content that can be identified."""

    CODE = "code"
    LOGS = "logs"
    PROSE = "prose"
    TRANSCRIPT = "transcript"
    CHAT = "chat"
    LEGAL = "legal"
    ACADEMIC = "academic"
    UNKNOWN = "unknown"


class ContentIdentifier:
    """Identifies the type of input content for appropriate compression strategies."""

    # Patterns for different content types
    CODE_PATTERNS = [
        r"def\s+\w+\s*\(",  # Python function
        r"class\s+\w+\s*[:\(]",  # Python class
        r"import\s+\w+",  # Python import
        r"from\s+\w+\s+import",  # Python from import
        r"function\s+\w+\s*\(",  # JavaScript function
        r"const\s+\w+\s*=",  # JavaScript const
        r"public\s+class\s+\w+",  # Java class
        r"#include\s*<",  # C/C++ include
    ]

    LOG_PATTERNS = [
        r"\[\d{4}-\d{2}-\d{2}",  # Date in logs [2025-11-23]
        r"\d{2}:\d{2}:\d{2}",  # Time in logs 03:33:25
        r"\b(ERROR|WARN|INFO|DEBUG|TRACE)\b",  # Log levels
        r"\b(Exception|Error|Failed|Success)\b",  # Common log terms
    ]

    TRANSCRIPT_PATTERNS = [
        r"^[A-Z][a-z]+\s*[A-Z][a-z]+\s*:",  # Speaker: John Smith:
        r"^[A-Z]+\s*:",  # Speaker: JS:
        r"\b(um|uh|like|you know|I mean)\b",  # Filler words
        r"\b(meeting|agenda|action item|follow up)\b",  # Meeting terms
    ]

    CHAT_PATTERNS = [
        r"^\[\d{2}:\d{2}\]",  # [03:33] timestamp
        r"^<\w+>",  # <username> format
        r"\b(lol|lmao|brb|btw|imo|imho)\b",  # Chat abbreviations
        r"^@\w+",  # @mentions
    ]

    LEGAL_PATTERNS = [
        r"\b(whereas|hereby|herein|thereof|pursuant)\b",  # Legal terms
        r"\b(agreement|contract|party|parties|clause)\b",  # Legal document terms
        r"Section\s+\d+",  # Section references
        r"Article\s+[IVX]+",  # Article references
    ]

    ACADEMIC_PATTERNS = [
        r"\b(abstract|introduction|methodology|results|conclusion)\b",  # Paper sections
        r"\[\d+\]",  # Citations [1]
        r"\b(et al\.|ibid\.|op\. cit\.)\b",  # Academic abbreviations
        r"\b(hypothesis|theorem|lemma|corollary)\b",  # Academic terms
    ]

    def __init__(self):
        """Initialize content identifier."""
        self._compiled_patterns: Dict[ContentType, List[re.Pattern]] = {
            ContentType.CODE: [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.CODE_PATTERNS],
            ContentType.LOGS: [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.LOG_PATTERNS],
            ContentType.TRANSCRIPT: [re.compile(p, re.MULTILINE) for p in self.TRANSCRIPT_PATTERNS],
            ContentType.CHAT: [re.compile(p, re.MULTILINE) for p in self.CHAT_PATTERNS],
            ContentType.LEGAL: [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.LEGAL_PATTERNS],
            ContentType.ACADEMIC: [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.ACADEMIC_PATTERNS],
        }

    def identify(self, content: str) -> ContentType:
        """Identify the type of content.

        Args:
            content: Text content to identify

        Returns:
            ContentType enum value
        """
        if not content or not content.strip():
            return ContentType.UNKNOWN

        # Count matches for each content type
        scores: Dict[ContentType, int] = {}

        for content_type, patterns in self._compiled_patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(content)
                score += len(matches)
            scores[content_type] = score

        # Get the type with highest score
        max_score = max(scores.values())

        # If no strong signal, default to PROSE
        if max_score == 0:
            return ContentType.PROSE

        # Return the type with highest score
        for content_type, score in scores.items():
            if score == max_score:
                return content_type

        return ContentType.PROSE

    def get_confidence(self, content: str, content_type: ContentType) -> float:
        """Get confidence score for a specific content type.

        Args:
            content: Text content
            content_type: Type to check confidence for

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if content_type not in self._compiled_patterns:
            return 0.0

        patterns = self._compiled_patterns[content_type]
        total_matches = 0

        for pattern in patterns:
            matches = pattern.findall(content)
            total_matches += len(matches)

        # Normalize by content length (lines)
        lines = content.count("\n") + 1
        confidence = min(1.0, total_matches / max(1, lines / 10))

        return confidence
