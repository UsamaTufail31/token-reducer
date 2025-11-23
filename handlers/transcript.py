"""Meeting transcript compression handler."""

import re
from typing import Dict, List, Tuple


class TranscriptHandler:
    """Specialized handler for meeting transcript compression."""

    def __init__(self):
        """Initialize transcript handler."""
        self.filler_words = {
            "um",
            "uh",
            "er",
            "ah",
            "like",
            "you know",
            "I mean",
            "sort of",
            "kind of",
            "basically",
            "actually",
            "literally",
        }

        self.speaker_pattern = re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*:", re.MULTILINE)

    def compress_transcript(self, transcript: str) -> Tuple[str, Dict[str, str]]:
        """Compress meeting transcript.

        Args:
            transcript: Transcript text

        Returns:
            Tuple of (compressed transcript, speaker mapping)
        """
        # Normalize speakers
        transcript, speaker_map = self._normalize_speakers(transcript)

        # Remove filler words
        transcript = self._remove_fillers(transcript)

        # Remove repeated acknowledgments
        transcript = self._remove_acknowledgments(transcript)

        return transcript, speaker_map

    def _normalize_speakers(self, transcript: str) -> Tuple[str, Dict[str, str]]:
        """Normalize speaker names to abbreviations.

        Args:
            transcript: Transcript text

        Returns:
            Tuple of (normalized transcript, speaker mapping)
        """
        # Find all speakers
        speakers = set(self.speaker_pattern.findall(transcript))

        # Create abbreviations
        speaker_map = {}
        for speaker in speakers:
            # Create abbreviation from initials
            parts = speaker.split()
            abbrev = "".join([p[0].upper() for p in parts])
            speaker_map[abbrev] = speaker

            # Replace in transcript
            transcript = transcript.replace(f"{speaker}:", f"{abbrev}:")

        return transcript, speaker_map

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words from text.

        Args:
            text: Text to process

        Returns:
            Text without fillers
        """
        for filler in self.filler_words:
            # Remove filler with word boundaries
            pattern = r"\b" + re.escape(filler) + r"\b[,\s]*"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)

        return text

    def _remove_acknowledgments(self, text: str) -> str:
        """Remove simple acknowledgments and agreements.

        Args:
            text: Text to process

        Returns:
            Text without acknowledgments
        """
        # Patterns for acknowledgments
        ack_patterns = [
            r"^[A-Z]+:\s*(okay|ok|alright|sure|right|yeah|yes|got it|sounds good)[.,\s]*$",
            r"^[A-Z]+:\s*(thanks|thank you|great|perfect|excellent)[.,\s]*$",
        ]

        lines = text.split("\n")
        filtered_lines = []

        for line in lines:
            is_ack = False
            for pattern in ack_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    is_ack = True
                    break

            if not is_ack and line.strip():
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def extract_action_items(self, transcript: str) -> List[str]:
        """Extract action items from transcript.

        Args:
            transcript: Transcript text

        Returns:
            List of action items
        """
        action_patterns = [
            r"action item[:\s]+(.+?)(?:\n|$)",
            r"(?:will|should|need to|must)\s+(.+?)(?:\n|\.)",
            r"(?:TODO|FIXME|ACTION)[:\s]+(.+?)(?:\n|$)",
        ]

        action_items = []
        for pattern in action_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            action_items.extend(matches)

        return action_items

    def extract_decisions(self, transcript: str) -> List[str]:
        """Extract decisions from transcript.

        Args:
            transcript: Transcript text

        Returns:
            List of decisions
        """
        decision_patterns = [
            r"(?:decided|agreed|concluded)\s+(?:that|to)\s+(.+?)(?:\n|\.)",
            r"decision[:\s]+(.+?)(?:\n|$)",
        ]

        decisions = []
        for pattern in decision_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            decisions.extend(matches)

        return decisions

    def summarize_meeting(self, transcript: str) -> str:
        """Create a summary of the meeting.

        Args:
            transcript: Transcript text

        Returns:
            Meeting summary
        """
        # Extract key information
        action_items = self.extract_action_items(transcript)
        decisions = self.extract_decisions(transcript)

        summary_parts = ["Meeting Summary\n"]

        if decisions:
            summary_parts.append("Decisions:")
            for i, decision in enumerate(decisions[:5], 1):
                summary_parts.append(f"  {i}. {decision.strip()}")
            summary_parts.append("")

        if action_items:
            summary_parts.append("Action Items:")
            for i, item in enumerate(action_items[:5], 1):
                summary_parts.append(f"  {i}. {item.strip()}")

        return "\n".join(summary_parts)
