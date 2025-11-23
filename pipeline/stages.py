"""Multi-stage pipeline stages for advanced compression."""

from typing import List, Tuple

from ..types import PipelineState
from .content_identifier import ContentIdentifier, ContentType
from .pass_base import CompressionPass
from .segmenter import Segmenter


class IdentificationStage(CompressionPass):
    """Stage 1: Identify content type."""

    def __init__(self):
        """Initialize identification stage."""
        self.identifier = ContentIdentifier()

    def execute(self, state: PipelineState) -> Tuple[str, List[str]]:
        """Execute content identification.

        Args:
            state: Current pipeline state

        Returns:
            Tuple of (content, warnings)
        """
        content_type = self.identifier.identify(state.current_content)

        # Store in metadata
        if not hasattr(state, "metadata"):
            state.metadata = {}
        state.metadata["content_type"] = content_type.value
        state.metadata["content_type_confidence"] = self.identifier.get_confidence(state.current_content, content_type)

        warnings = []
        if content_type == ContentType.UNKNOWN:
            warnings.append("Could not identify content type, using default compression")

        return state.current_content, warnings


class SegmentationStage(CompressionPass):
    """Stage 2: Segment content into processing units."""

    def __init__(self):
        """Initialize segmentation stage."""
        self.segmenter = Segmenter()

    def execute(self, state: PipelineState) -> Tuple[str, List[str]]:
        """Execute content segmentation.

        Args:
            state: Current pipeline state

        Returns:
            Tuple of (content, warnings)
        """
        # Get content type from metadata
        content_type = getattr(state, "metadata", {}).get("content_type", "prose")

        # Segment based on content type
        if content_type == "code":
            segments = self.segmenter.segment_code(state.current_content)
        elif content_type == "logs":
            segments = self.segmenter.segment_logs(state.current_content)
        else:
            segments = self.segmenter.segment_text(state.current_content, level="sentence")

        # Store segments in metadata
        if not hasattr(state, "metadata"):
            state.metadata = {}
        state.metadata["segments"] = segments
        state.metadata["segment_count"] = len(segments)

        warnings = []
        if len(segments) == 0:
            warnings.append("No segments identified in content")

        return state.current_content, warnings


class RedundancyRemovalStage(CompressionPass):
    """Stage 3: Remove structural redundancies."""

    def execute(self, state: PipelineState) -> Tuple[str, List[str]]:
        """Execute redundancy removal.

        Args:
            state: Current pipeline state

        Returns:
            Tuple of (content, warnings)
        """
        content = state.current_content
        warnings = []

        # Get content type
        content_type = getattr(state, "metadata", {}).get("content_type", "prose")

        if content_type == "logs":
            content, log_warnings = self._remove_log_redundancy(content)
            warnings.extend(log_warnings)
        else:
            content, text_warnings = self._remove_text_redundancy(content)
            warnings.extend(text_warnings)

        return content, warnings

    def _remove_log_redundancy(self, content: str) -> Tuple[str, List[str]]:
        """Remove redundancy from log content.

        Args:
            content: Log content

        Returns:
            Tuple of (compressed content, warnings)
        """
        import re
        from collections import Counter

        lines = content.split("\n")
        line_counts = Counter(lines)

        # Find repeated lines
        compressed_lines = []
        seen = set()

        for line in lines:
            if line.strip() and line_counts[line] > 3:
                # Repeated line
                if line not in seen:
                    compressed_lines.append(f"{line} (repeated {line_counts[line]}x)")
                    seen.add(line)
            else:
                compressed_lines.append(line)

        warnings = []
        if len(compressed_lines) < len(lines):
            warnings.append(f"Collapsed {len(lines) - len(compressed_lines)} repeated log lines")

        return "\n".join(compressed_lines), warnings

    def _remove_text_redundancy(self, content: str) -> Tuple[str, List[str]]:
        """Remove redundancy from text content.

        Args:
            content: Text content

        Returns:
            Tuple of (compressed content, warnings)
        """
        import re

        # Remove filler phrases
        filler_patterns = [
            r"\b(okay|ok|alright|sure|right|yeah|yes|no|well|so|like|you know|I mean|um|uh|er|ah)\b[,\s]*",
        ]

        original_length = len(content)
        for pattern in filler_patterns:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE)

        # Clean up extra whitespace
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"\s+([.!?,;:])", r"\1", content)

        warnings = []
        if len(content) < original_length:
            warnings.append(f"Removed {original_length - len(content)} characters of filler content")

        return content.strip(), warnings
