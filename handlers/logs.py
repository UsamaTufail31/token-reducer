"""Log file compression handler."""

import re
from collections import Counter
from typing import Dict, List, Tuple


class LogHandler:
    """Specialized handler for log file compression."""

    def __init__(self):
        """Initialize log handler."""
        self.log_pattern = re.compile(
            r"^(\[?\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2})",
            re.MULTILINE,
        )
        self.level_pattern = re.compile(r"\b(ERROR|WARN|INFO|DEBUG|TRACE)\b")

    def compress_logs(self, logs: str, collapse_threshold: int = 3) -> Tuple[str, Dict[str, int]]:
        """Compress log content.

        Args:
            logs: Log content to compress
            collapse_threshold: Number of repetitions before collapsing

        Returns:
            Tuple of (compressed logs, statistics)
        """
        lines = logs.split("\n")

        # Normalize timestamps
        normalized_lines = []
        for line in lines:
            normalized = self._normalize_timestamp(line)
            normalized_lines.append(normalized)

        # Find repeated patterns
        line_counts = Counter(normalized_lines)

        # Compress
        compressed_lines = []
        seen = set()
        stats = {"original_lines": len(lines), "collapsed_lines": 0, "removed_lines": 0}

        for line in normalized_lines:
            if not line.strip():
                continue

            count = line_counts[line]

            if count >= collapse_threshold:
                if line not in seen:
                    compressed_lines.append(f"{line} (repeated {count}x)")
                    seen.add(line)
                    stats["collapsed_lines"] += count - 1
            else:
                compressed_lines.append(line)

        stats["final_lines"] = len(compressed_lines)

        return "\n".join(compressed_lines), stats

    def _normalize_timestamp(self, line: str) -> str:
        """Normalize or remove timestamps from log line.

        Args:
            line: Log line

        Returns:
            Normalized line
        """
        # Replace full timestamp with [TIMESTAMP]
        line = re.sub(
            r"\[?\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\]?",
            "[TS]",
            line,
        )

        # Replace time-only timestamp
        line = re.sub(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?", "[TS]", line)

        return line

    def extract_errors(self, logs: str) -> List[str]:
        """Extract only error and warning lines.

        Args:
            logs: Log content

        Returns:
            List of error/warning lines
        """
        lines = logs.split("\n")
        error_lines = []

        for line in lines:
            if re.search(r"\b(ERROR|WARN|FATAL|CRITICAL)\b", line, re.IGNORECASE):
                error_lines.append(line)

        return error_lines

    def summarize_logs(self, logs: str) -> str:
        """Create a summary of log content.

        Args:
            logs: Log content

        Returns:
            Summary string
        """
        lines = logs.split("\n")

        # Count log levels
        level_counts = Counter()
        for line in lines:
            match = self.level_pattern.search(line)
            if match:
                level_counts[match.group(1)] += 1

        # Extract unique error messages
        error_lines = self.extract_errors(logs)
        unique_errors = list(set(error_lines))[:5]  # Top 5 unique errors

        # Build summary
        summary_parts = [f"Total log lines: {len(lines)}"]

        if level_counts:
            summary_parts.append("Log levels:")
            for level, count in level_counts.most_common():
                summary_parts.append(f"  {level}: {count}")

        if unique_errors:
            summary_parts.append("\nSample errors:")
            for error in unique_errors:
                summary_parts.append(f"  - {error[:100]}...")

        return "\n".join(summary_parts)
