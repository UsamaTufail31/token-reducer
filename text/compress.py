"""Semantic compression pass for token reduction."""

import re
from typing import Dict, List, Set, Tuple

from .normalize import segment_sentences


def extract_entities(text: str) -> Set[str]:
    """Extract named entities (simple pattern-based approach).

    Args:
        text: Input text

    Returns:
        Set of extracted entities
    """
    entities = set()

    # Capitalized words (potential proper nouns)
    # Match sequences of capitalized words
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    entities.update(proper_nouns)

    # Organizations (Inc., Corp., Ltd., etc.)
    orgs = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Co)\b\.?", text
    )
    entities.update(orgs)

    return entities


def extract_numbers(text: str) -> List[Tuple[str, str]]:
    """Extract numerical data (numbers, dates, amounts).

    Args:
        text: Input text

    Returns:
        List of (number, context) tuples
    """
    numbers = []

    # Dates (various formats)
    date_patterns = [
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",  # MM/DD/YYYY
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",  # YYYY-MM-DD
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    ]

    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            numbers.append((match.group(), "date"))

    # Currency amounts
    currency_pattern = r"[$£€¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?"
    matches = re.finditer(currency_pattern, text)
    for match in matches:
        numbers.append((match.group(), "currency"))

    # Percentages
    percentage_pattern = r"\b\d+(?:\.\d+)?%"
    matches = re.finditer(percentage_pattern, text)
    for match in matches:
        numbers.append((match.group(), "percentage"))

    # General numbers
    number_pattern = r"\b\d+(?:,\d{3})*(?:\.\d+)?\b"
    matches = re.finditer(number_pattern, text)
    for match in matches:
        # Skip if already captured as date/currency/percentage
        if not any(match.group() in num[0] for num in numbers):
            numbers.append((match.group(), "number"))

    return numbers


def identify_topic_sentences(sentences: List[str]) -> Set[int]:
    """Identify topic sentences (first sentence of paragraphs, sentences with key markers).

    Args:
        sentences: List of sentences

    Returns:
        Set of indices of topic sentences
    """
    topic_indices = set()

    # First sentence is often a topic sentence
    if sentences:
        topic_indices.add(0)

    # Sentences with topic markers
    topic_markers = [
        r"^(the main|the primary|the key|the most important)",
        r"^(first|second|third|finally|in conclusion)",
        r"^(overall|in summary|to summarize)",
        r"\b(therefore|thus|hence|consequently)\b",
    ]

    patterns = [re.compile(marker, re.IGNORECASE) for marker in topic_markers]

    for i, sentence in enumerate(sentences):
        for pattern in patterns:
            if pattern.search(sentence):
                topic_indices.add(i)
                break

    return topic_indices


def identify_factual_statements(sentences: List[str]) -> Set[int]:
    """Identify sentences containing factual statements.

    Args:
        sentences: List of sentences

    Returns:
        Set of indices of factual sentences
    """
    factual_indices = set()

    # Factual markers
    factual_markers = [
        r"\b(is|are|was|were|has been|have been)\b",
        r"\b(according to|based on|research shows|studies show)\b",
        r"\b(fact|data|evidence|statistics|research)\b",
        r"\b\d+%",  # Contains percentages
        r"\$\d+",  # Contains money
    ]

    patterns = [re.compile(marker, re.IGNORECASE) for marker in factual_markers]

    for i, sentence in enumerate(sentences):
        # Check for factual markers
        marker_count = sum(1 for pattern in patterns if pattern.search(sentence))

        # Sentences with multiple factual markers are likely factual
        if marker_count >= 2:
            factual_indices.add(i)

        # Sentences with numbers/dates are often factual
        if extract_numbers(sentence):
            factual_indices.add(i)

    return factual_indices


def identify_instructions(sentences: List[str]) -> Set[int]:
    """Identify instructional sentences.

    Args:
        sentences: List of sentences

    Returns:
        Set of indices of instructional sentences
    """
    instruction_indices = set()

    # Instruction markers (imperative verbs, modal verbs)
    instruction_markers = [
        r"^(please|make sure|ensure|verify|check|confirm)",
        r"^(do not|don't|never|always|must|should|need to)",
        r"^(click|select|choose|enter|type|press|open|close)",
        r"^(follow|complete|submit|review|update|delete|create)",
    ]

    patterns = [re.compile(marker, re.IGNORECASE) for marker in instruction_markers]

    for i, sentence in enumerate(sentences):
        for pattern in patterns:
            if pattern.search(sentence):
                instruction_indices.add(i)
                break

    return instruction_indices


def compact_phrasing(text: str) -> str:
    """Transform verbose phrasing to compact equivalents.

    Args:
        text: Input text

    Returns:
        Text with compact phrasing
    """
    # Verbose to compact replacements
    replacements = {
        r"\bin order to\b": "to",
        r"\bdue to the fact that\b": "because",
        r"\bin the event that\b": "if",
        r"\bat this point in time\b": "now",
        r"\bat the present time\b": "now",
        r"\bin the near future\b": "soon",
        r"\bprior to\b": "before",
        r"\bsubsequent to\b": "after",
        r"\bin spite of the fact that\b": "although",
        r"\bfor the purpose of\b": "for",
        r"\bin the process of\b": "while",
        r"\bhas the ability to\b": "can",
        r"\bis able to\b": "can",
        r"\bmake a decision\b": "decide",
        r"\bmake a determination\b": "determine",
        r"\bcome to a conclusion\b": "conclude",
        r"\btake into consideration\b": "consider",
    }

    for verbose, compact in replacements.items():
        text = re.sub(verbose, compact, text, flags=re.IGNORECASE)

    return text


def reduce_modifiers(text: str, aggressive: bool = False) -> str:
    """Remove or reduce adjectives and adverbs.

    Args:
        text: Input text
        aggressive: Whether to aggressively remove modifiers

    Returns:
        Text with reduced modifiers
    """
    if aggressive:
        # Remove common intensifiers
        intensifiers = [
            r"\b(very|extremely|incredibly|absolutely|totally|completely)\s+",
            r"\b(quite|rather|fairly|pretty|somewhat)\s+",
            r"\b(really|actually|basically|essentially|literally)\s+",
        ]

        for pattern in intensifiers:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove filler adverbs
        filler_adverbs = [
            r"\b(just|simply|merely|only)\s+",
            r"\b(clearly|obviously|evidently|apparently)\s+",
        ]

        for pattern in filler_adverbs:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text


def compress_semantically(
    text: str,
    preserve_entities: bool = True,
    preserve_numbers: bool = True,
    preserve_facts: bool = True,
    preserve_instructions: bool = True,
    aggressive: bool = False,
) -> str:
    """Apply semantic compression to text.

    Args:
        text: Input text
        preserve_entities: Whether to preserve named entities
        preserve_numbers: Whether to preserve numerical data
        preserve_facts: Whether to preserve factual statements
        preserve_instructions: Whether to preserve instructions
        aggressive: Whether to use aggressive compression

    Returns:
        Semantically compressed text
    """
    # Segment into sentences
    sentences = segment_sentences(text)

    if not sentences:
        return text

    # Identify important sentences
    important_indices = set()

    # Always preserve topic sentences
    important_indices.update(identify_topic_sentences(sentences))

    if preserve_facts:
        important_indices.update(identify_factual_statements(sentences))

    if preserve_instructions:
        important_indices.update(identify_instructions(sentences))

    # If preserving entities/numbers, keep sentences containing them
    if preserve_entities or preserve_numbers:
        for i, sentence in enumerate(sentences):
            if preserve_entities and extract_entities(sentence):
                important_indices.add(i)
            if preserve_numbers and extract_numbers(sentence):
                important_indices.add(i)

    # Keep important sentences
    compressed_sentences = [
        sentences[i] for i in sorted(important_indices) if i < len(sentences)
    ]

    # If too aggressive, ensure we keep at least some content
    if not compressed_sentences and sentences:
        # Keep first and last sentences at minimum
        compressed_sentences = [sentences[0]]
        if len(sentences) > 1:
            compressed_sentences.append(sentences[-1])

    # Rejoin sentences
    result = " ".join(compressed_sentences)

    # Apply phrasing transformations
    result = compact_phrasing(result)

    # Reduce modifiers
    result = reduce_modifiers(result, aggressive)

    return result


class SemanticCompressor:
    """Semantic compression pass for compression pipeline."""

    def __init__(
        self,
        preserve_entities: bool = True,
        preserve_numbers: bool = True,
        preserve_facts: bool = True,
        preserve_instructions: bool = True,
        aggressive: bool = False,
    ) -> None:
        """Initialize semantic compressor.

        Args:
            preserve_entities: Whether to preserve named entities
            preserve_numbers: Whether to preserve numerical data
            preserve_facts: Whether to preserve factual statements
            preserve_instructions: Whether to preserve instructions
            aggressive: Whether to use aggressive compression
        """
        self.preserve_entities = preserve_entities
        self.preserve_numbers = preserve_numbers
        self.preserve_facts = preserve_facts
        self.preserve_instructions = preserve_instructions
        self.aggressive = aggressive

    def process(self, text: str) -> str:
        """Process text through semantic compression.

        Args:
            text: Input text

        Returns:
            Semantically compressed text
        """
        return compress_semantically(
            text,
            self.preserve_entities,
            self.preserve_numbers,
            self.preserve_facts,
            self.preserve_instructions,
            self.aggressive,
        )

    def should_run(self, state: "PipelineState") -> bool:  # type: ignore
        """Check if pass should run."""
        return True

    @property
    def name(self) -> str:
        """Get pass name."""
        return "compress"
