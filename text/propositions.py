"""Proposition extraction for sentence simplification."""

import re
from typing import List, Optional


class PropositionExtractor:
    """Extracts core propositions from complex sentences."""

    def __init__(self):
        """Initialize proposition extractor."""
        # Common filler phrases to remove
        self.filler_phrases = [
            r"due to the fact that",
            r"in order to",
            r"for the purpose of",
            r"with regard to",
            r"in relation to",
            r"in the event that",
            r"in spite of the fact that",
            r"despite the fact that",
            r"it is important to note that",
            r"it should be noted that",
            r"it is worth mentioning that",
        ]

        # Simplification patterns
        self.simplifications = [
            (r"due to the fact that", "because"),
            (r"in order to", "to"),
            (r"for the purpose of", "to"),
            (r"with regard to", "about"),
            (r"in relation to", "about"),
            (r"in the event that", "if"),
            (r"in spite of the fact that", "although"),
            (r"despite the fact that", "although"),
        ]

    def extract_proposition(self, sentence: str) -> str:
        """Extract core proposition from a sentence.

        Args:
            sentence: Sentence to simplify

        Returns:
            Simplified proposition
        """
        result = sentence

        # Apply simplifications
        for pattern, replacement in self.simplifications:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Remove redundant phrases
        result = self._remove_redundant_phrases(result)

        # Extract core subject-verb-object if possible
        result = self._extract_core_svo(result)

        return result.strip()

    def _remove_redundant_phrases(self, text: str) -> str:
        """Remove redundant phrases from text.

        Args:
            text: Text to process

        Returns:
            Text with redundant phrases removed
        """
        # Remove "it is X that" constructions
        text = re.sub(r"it is \w+ that\s+", "", text, flags=re.IGNORECASE)

        # Remove "there is/are" constructions
        text = re.sub(r"there (?:is|are|was|were)\s+", "", text, flags=re.IGNORECASE)

        # Remove excessive adverbs
        text = re.sub(r"\b(very|really|quite|rather|extremely|incredibly)\s+", "", text, flags=re.IGNORECASE)

        return text

    def _extract_core_svo(self, sentence: str) -> str:
        """Extract core subject-verb-object structure.

        Args:
            sentence: Sentence to process

        Returns:
            Core SVO structure
        """
        # This is a simplified heuristic approach
        # A full implementation would use dependency parsing

        # Remove relative clauses
        sentence = re.sub(r",\s*which\s+[^,]+,", "", sentence)
        sentence = re.sub(r",\s*who\s+[^,]+,", "", sentence)

        # Remove parenthetical expressions
        sentence = re.sub(r"\([^)]+\)", "", sentence)

        # Remove prepositional phrases at the end (heuristic)
        sentence = re.sub(r"\s+(?:in|on|at|by|with|from|to)\s+the\s+\w+\s*$", "", sentence)

        return sentence

    def extract_propositions_from_text(self, text: str) -> List[str]:
        """Extract propositions from multiple sentences.

        Args:
            text: Text containing multiple sentences

        Returns:
            List of extracted propositions
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+\s+", text)

        propositions = []
        for sentence in sentences:
            if sentence.strip():
                prop = self.extract_proposition(sentence)
                if prop:
                    propositions.append(prop)

        return propositions

    def simplify_text(self, text: str, join_with: str = ". ") -> str:
        """Simplify entire text by extracting propositions.

        Args:
            text: Text to simplify
            join_with: String to join propositions with

        Returns:
            Simplified text
        """
        propositions = self.extract_propositions_from_text(text)
        return join_with.join(propositions)
