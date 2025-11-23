"""Text normalization pass for token reduction."""

import re
from typing import List

# Filler phrases to remove
FILLER_PHRASES = [
    r"\b(please note that|it should be noted that|it is important to note that)\b",
    r"\b(as you (may |can )?know|as mentioned (before|earlier|previously))\b",
    r"\b(in other words|to put it (simply|differently|another way))\b",
    r"\b(basically|essentially|actually|literally)\b",
    r"\b(kind of|sort of|type of)\b",
    r"^(hi|hello|hey|dear|greetings)[,\s]",
    r"(thanks|thank you|regards|best regards|sincerely)[,\s]*$",
    r"\b(just|really|very|quite|rather|somewhat)\b",
]

# Compile regex patterns
FILLER_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in FILLER_PHRASES]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    # Replace multiple newlines with double newline (preserve paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove trailing/leading whitespace from lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def remove_html_markup(text: str) -> str:
    """Remove HTML/XML tags and markup.

    Args:
        text: Input text

    Returns:
        Text with HTML removed
    """
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove script and style tags with content
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode common HTML entities
    html_entities = {
        "&nbsp;": " ",
        "&lt;": "<",
        "&gt;": ">",
        "&amp;": "&",
        "&quot;": '"',
        "&#39;": "'",
        "&apos;": "'",
    }

    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    return text


def deduplicate_punctuation(text: str) -> str:
    """Remove repeated punctuation.

    Args:
        text: Input text

    Returns:
        Text with deduplicated punctuation
    """
    # Reduce multiple exclamation marks
    text = re.sub(r"!{2,}", "!", text)

    # Reduce multiple question marks
    text = re.sub(r"\?{2,}", "?", text)

    # Preserve intentional ellipsis (...) but reduce longer sequences
    text = re.sub(r"\.{4,}", "...", text)

    # Remove multiple commas
    text = re.sub(r",{2,}", ",", text)

    # Remove multiple semicolons
    text = re.sub(r";{2,}", ";", text)

    # Remove multiple colons
    text = re.sub(r":{2,}", ":", text)

    return text


def standardize_quotes(text: str) -> str:
    """Standardize quote characters.

    Args:
        text: Input text

    Returns:
        Text with standardized quotes
    """
    # Replace curly quotes with straight quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")

    # Replace guillemets
    text = text.replace("«", '"').replace("»", '"')

    # Replace backticks used as quotes
    text = re.sub(r"`([^`]+)`", r"'\1'", text)

    return text


def segment_sentences(text: str) -> List[str]:
    """Segment text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Simple sentence segmentation
    # Split on period, exclamation, question mark followed by space and capital
    sentences = re.split(r"([.!?])\s+(?=[A-Z])", text)

    # Rejoin punctuation with sentences
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i + 1])
        else:
            result.append(sentences[i])

    # Add last sentence if exists
    if len(sentences) % 2 == 1:
        result.append(sentences[-1])

    # Filter empty sentences
    result = [s.strip() for s in result if s.strip()]

    return result


def remove_filler_content(text: str) -> str:
    """Remove non-informative filler phrases.

    Args:
        text: Input text

    Returns:
        Text with filler removed
    """
    for pattern in FILLER_PATTERNS:
        text = pattern.sub("", text)

    # Clean up any double spaces created
    text = re.sub(r" {2,}", " ", text)

    # Clean up spaces before punctuation
    text = re.sub(r" ([.,;:!?])", r"\1", text)

    return text.strip()


def normalize_text(text: str, preserve_structure: bool = True) -> str:
    """Apply all normalization steps to text.

    Args:
        text: Input text
        preserve_structure: Whether to preserve paragraph structure

    Returns:
        Normalized text
    """
    # Remove HTML/markup
    text = remove_html_markup(text)

    # Standardize quotes
    text = standardize_quotes(text)

    # Deduplicate punctuation
    text = deduplicate_punctuation(text)

    # Normalize whitespace
    text = normalize_whitespace(text)

    # Remove filler content
    text = remove_filler_content(text)

    # Final whitespace cleanup
    text = normalize_whitespace(text)

    return text


class TextNormalizer:
    """Text normalization pass for compression pipeline."""

    def __init__(self, preserve_structure: bool = True) -> None:
        """Initialize normalizer.

        Args:
            preserve_structure: Whether to preserve paragraph boundaries
        """
        self.preserve_structure = preserve_structure

    def process(self, text: str) -> str:
        """Process text through normalization.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        return normalize_text(text, self.preserve_structure)

    def should_run(self, state: "PipelineState") -> bool:  # type: ignore
        """Check if pass should run."""
        return True

    @property
    def name(self) -> str:
        """Get pass name."""
        return "normalize"
