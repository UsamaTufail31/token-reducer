"""Content segmentation for structural processing."""

import re
from typing import List, Optional


class Segment:
    """Represents a segment of content."""

    def __init__(self, content: str, segment_type: str, metadata: Optional[dict] = None):
        """Initialize segment.

        Args:
            content: Segment content
            segment_type: Type of segment (sentence, paragraph, function, etc.)
            metadata: Optional metadata about the segment
        """
        self.content = content
        self.segment_type = segment_type
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Segment(type={self.segment_type}, length={len(self.content)})"


class Segmenter:
    """Segments content into logical processing units."""

    def __init__(self):
        """Initialize segmenter."""
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r"([.!?]+)\s+")

    def segment_text(self, text: str, level: str = "sentence") -> List[Segment]:
        """Segment text into sentences or paragraphs.

        Args:
            text: Text to segment
            level: Segmentation level ('sentence' or 'paragraph')

        Returns:
            List of Segment objects
        """
        if level == "paragraph":
            return self._segment_paragraphs(text)
        else:
            return self._segment_sentences(text)

    def _segment_sentences(self, text: str) -> List[Segment]:
        """Segment text into sentences.

        Args:
            text: Text to segment

        Returns:
            List of sentence segments
        """
        segments = []

        # Split by sentence boundaries
        sentences = self.sentence_endings.split(text)

        # Reconstruct sentences with their punctuation
        current_sentence = ""
        for i, part in enumerate(sentences):
            if i % 2 == 0:
                # This is the sentence content
                current_sentence = part
            else:
                # This is the punctuation
                current_sentence += part
                if current_sentence.strip():
                    segments.append(
                        Segment(
                            content=current_sentence.strip(),
                            segment_type="sentence",
                            metadata={"index": len(segments)},
                        )
                    )
                current_sentence = ""

        # Handle last sentence if no punctuation
        if current_sentence.strip():
            segments.append(
                Segment(
                    content=current_sentence.strip(),
                    segment_type="sentence",
                    metadata={"index": len(segments)},
                )
            )

        return segments

    def _segment_paragraphs(self, text: str) -> List[Segment]:
        """Segment text into paragraphs.

        Args:
            text: Text to segment

        Returns:
            List of paragraph segments
        """
        segments = []

        # Split by double newlines
        paragraphs = re.split(r"\n\s*\n", text)

        for i, para in enumerate(paragraphs):
            if para.strip():
                segments.append(
                    Segment(
                        content=para.strip(),
                        segment_type="paragraph",
                        metadata={"index": i},
                    )
                )

        return segments

    def segment_code(self, code: str, language: str = "python") -> List[Segment]:
        """Segment code into functions/classes.

        Args:
            code: Code to segment
            language: Programming language

        Returns:
            List of code segments
        """
        if language == "python":
            return self._segment_python_code(code)
        else:
            # Fallback: treat as single segment
            return [Segment(content=code, segment_type="code_block", metadata={"language": language})]

    def _segment_python_code(self, code: str) -> List[Segment]:
        """Segment Python code into functions and classes.

        Args:
            code: Python code to segment

        Returns:
            List of code segments
        """
        segments = []
        lines = code.split("\n")

        current_block = []
        current_type = None
        indent_level = 0

        for i, line in enumerate(lines):
            # Check for function or class definition
            if re.match(r"^(def|class)\s+\w+", line):
                # Save previous block if exists
                if current_block:
                    segments.append(
                        Segment(
                            content="\n".join(current_block),
                            segment_type=current_type or "code_block",
                            metadata={"start_line": i - len(current_block)},
                        )
                    )

                # Start new block
                current_block = [line]
                current_type = "function" if line.startswith("def") else "class"
                indent_level = len(line) - len(line.lstrip())

            elif current_block:
                # Check if still in the same block
                stripped = line.lstrip()
                if stripped and not line.startswith(" " * (indent_level + 1)) and not stripped.startswith("#"):
                    # End of current block
                    segments.append(
                        Segment(
                            content="\n".join(current_block),
                            segment_type=current_type or "code_block",
                            metadata={"start_line": i - len(current_block)},
                        )
                    )
                    current_block = [line]
                    current_type = "code_block"
                else:
                    current_block.append(line)
            else:
                current_block.append(line)
                current_type = "code_block"

        # Add final block
        if current_block:
            segments.append(
                Segment(
                    content="\n".join(current_block),
                    segment_type=current_type or "code_block",
                    metadata={"start_line": len(lines) - len(current_block)},
                )
            )

        return segments

    def segment_logs(self, logs: str) -> List[Segment]:
        """Segment logs into individual log entries.

        Args:
            logs: Log content to segment

        Returns:
            List of log entry segments
        """
        segments = []

        # Pattern for log entry start (timestamp or log level)
        log_start_pattern = re.compile(r"^(\[\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|\b(ERROR|WARN|INFO|DEBUG)\b)", re.MULTILINE)

        lines = logs.split("\n")
        current_entry = []

        for line in lines:
            if log_start_pattern.match(line):
                # Start of new log entry
                if current_entry:
                    segments.append(
                        Segment(
                            content="\n".join(current_entry),
                            segment_type="log_entry",
                            metadata={"index": len(segments)},
                        )
                    )
                current_entry = [line]
            else:
                # Continuation of current entry
                if current_entry or line.strip():
                    current_entry.append(line)

        # Add final entry
        if current_entry:
            segments.append(
                Segment(
                    content="\n".join(current_entry),
                    segment_type="log_entry",
                    metadata={"index": len(segments)},
                )
            )

        return segments
