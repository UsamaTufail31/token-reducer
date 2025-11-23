"""Final text repackaging for token efficiency."""

import re
from typing import List

from .normalize import segment_sentences


def shorten_sentences(text: str) -> str:
    """Use shorter sentence structures.

    Args:
        text: Input text

    Returns:
        Text with shortened sentences
    """
    # Remove redundant pronouns at sentence starts
    text = re.sub(r"(?<=\. )(It is|There is|There are)\s+", "", text)

    # Simplify "which is/are" constructions
    text = re.sub(r",?\s+which (is|are)\s+", " ", text)

    return text


def optimize_newlines(text: str) -> str:
    """Minimize unnecessary newlines.

    Args:
        text: Input text

    Returns:
        Text with optimized newlines
    """
    # Replace multiple newlines with single newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove newlines within sentences
    text = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", text)

    return text


def reduce_stopwords(text: str) -> str:
    """Reduce non-essential stopwords.

    Args:
        text: Input text

    Returns:
        Text with reduced stopwords
    """
    # Remove articles in non-critical positions
    # Be conservative to maintain readability
    text = re.sub(r"\b(a|an|the)\s+(?=\w+\s+(?:is|are|was|were))", "", text)

    return text


def convert_to_lists(text: str) -> str:
    """Convert paragraphs to bullet-point lists where appropriate.

    Args:
        text: Input text

    Returns:
        Text with list formatting
    """
    sentences = segment_sentences(text)

    if len(sentences) <= 2:
        # Too short for list conversion
        return text

    # Check if sentences follow a pattern (enumeration, similar structure)
    # Simple heuristic: if sentences start with similar words, convert to list
    first_words = [s.split()[0] if s.split() else "" for s in sentences]

    # If many sentences start with same word, likely a list
    from collections import Counter

    word_counts = Counter(first_words)
    most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0

    if most_common_count >= len(sentences) * 0.5:
        # Convert to compact list format
        return "\n".join(f"- {s}" for s in sentences)

    return text


def apply_consistent_formatting(text: str) -> str:
    """Apply consistent structural patterns.

    Args:
        text: Input text

    Returns:
        Consistently formatted text
    """
    # Standardize spacing around punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])(?=[A-Za-z])", r"\1 ", text)

    # Remove extra spaces
    text = re.sub(r" {2,}", " ", text)

    # Clean up line breaks
    text = re.sub(r"\n ", "\n", text)
    text = re.sub(r" \n", "\n", text)

    return text.strip()


def repackage_text(text: str, aggressive: bool = False) -> str:
    """Repackage text for token efficiency.

    Args:
        text: Input text
        aggressive: Whether to use aggressive repackaging

    Returns:
        Repackaged text
    """
    # Shorten sentences
    text = shorten_sentences(text)

    # Optimize newlines
    text = optimize_newlines(text)

    if aggressive:
        # Reduce stopwords
        text = reduce_stopwords(text)

        # Convert to lists if appropriate
        text = convert_to_lists(text)

    # Apply consistent formatting
    text = apply_consistent_formatting(text)

    return text


class TextRepackager:
    """Text repackaging pass for compression pipeline."""

    def __init__(self, aggressive: bool = False) -> None:
        """Initialize repackager.

        Args:
            aggressive: Whether to use aggressive repackaging
        """
        self.aggressive = aggressive

    def process(self, text: str) -> str:
        """Process text through repackaging.

        Args:
            text: Input text

        Returns:
            Repackaged text
        """
        return repackage_text(text, self.aggressive)

    def should_run(self, state: "PipelineState") -> bool:  # type: ignore
        """Check if pass should run."""
        return True

    @property
    def name(self) -> str:
        """Get pass name."""
        return "repack"
