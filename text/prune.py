"""Text redundancy removal pass for token reduction."""

import re
from typing import List, Set, Tuple

from .normalize import segment_sentences


def get_sentence_fingerprint(sentence: str) -> str:
    """Create a normalized fingerprint for duplicate detection.

    Args:
        sentence: Input sentence

    Returns:
        Normalized fingerprint
    """
    # Convert to lowercase
    fingerprint = sentence.lower()

    # Remove punctuation
    fingerprint = re.sub(r"[^\w\s]", "", fingerprint)

    # Normalize whitespace
    fingerprint = " ".join(fingerprint.split())

    return fingerprint


def remove_duplicate_sentences(sentences: List[str]) -> List[str]:
    """Remove exact and near-duplicate sentences.

    Args:
        sentences: List of sentences

    Returns:
        List with duplicates removed
    """
    seen_fingerprints: Set[str] = set()
    unique_sentences: List[str] = []

    for sentence in sentences:
        fingerprint = get_sentence_fingerprint(sentence)

        if fingerprint and fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_sentences.append(sentence)

    return unique_sentences


def detect_reformulations(sentences: List[str]) -> List[int]:
    """Detect sentences that are reformulations of previous sentences.

    Args:
        sentences: List of sentences

    Returns:
        List of indices to remove (reformulations)
    """
    reformulation_markers = [
        r"^(in other words|to put it (simply|differently|another way))",
        r"^(that is to say|which is to say|meaning|i\.e\.|i\.e,)",
        r"^(to clarify|to be clear|to explain)",
        r"^(what (this|that|I) mean is)",
    ]

    patterns = [re.compile(marker, re.IGNORECASE) for marker in reformulation_markers]

    indices_to_remove = []

    for i, sentence in enumerate(sentences):
        for pattern in patterns:
            if pattern.search(sentence):
                indices_to_remove.append(i)
                break

    return indices_to_remove


def detect_clarifying_sentences(sentences: List[str]) -> List[int]:
    """Detect sentences that merely clarify without adding new information.

    Args:
        sentences: List of sentences

    Returns:
        List of indices to remove
    """
    clarifying_markers = [
        r"^(as (I |we )?(mentioned|said|noted|stated) (before|earlier|previously|above))",
        r"^(again|once again|to reiterate|to repeat)",
        r"^(remember that|recall that|keep in mind that)",
        r"^(as you (may |can )?(know|see|recall))",
    ]

    patterns = [re.compile(marker, re.IGNORECASE) for marker in clarifying_markers]

    indices_to_remove = []

    for i, sentence in enumerate(sentences):
        for pattern in patterns:
            if pattern.search(sentence):
                indices_to_remove.append(i)
                break

    return indices_to_remove


def detect_verbose_explanations(sentences: List[str]) -> List[int]:
    """Detect overly verbose explanations of simple concepts.

    Args:
        sentences: List of sentences

    Returns:
        List of indices to remove
    """
    verbose_markers = [
        r"^(for example|for instance|such as)",
        r"^(this means that|what this means is)",
        r"^(in simple terms|simply put|to simplify)",
        r"\b(obviously|clearly|evidently|naturally)\b",
    ]

    patterns = [re.compile(marker, re.IGNORECASE) for marker in verbose_markers]

    indices_to_remove = []

    for i, sentence in enumerate(sentences):
        # Check for verbose markers
        has_marker = any(pattern.search(sentence) for pattern in patterns)

        # Check if sentence is very long (potential verbose explanation)
        word_count = len(sentence.split())
        is_very_long = word_count > 50

        if has_marker and is_very_long:
            indices_to_remove.append(i)

    return indices_to_remove


def calculate_sentence_similarity(sent1: str, sent2: str) -> float:
    """Calculate simple word-overlap similarity between sentences.

    Args:
        sent1: First sentence
        sent2: Second sentence

    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Tokenize and normalize
    words1 = set(get_sentence_fingerprint(sent1).split())
    words2 = set(get_sentence_fingerprint(sent2).split())

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def remove_semantic_redundancy(
    sentences: List[str], similarity_threshold: float = 0.7
) -> List[str]:
    """Remove semantically similar sentences.

    Args:
        sentences: List of sentences
        similarity_threshold: Threshold for considering sentences similar

    Returns:
        List with redundant sentences removed
    """
    if not sentences:
        return []

    # Keep track of which sentences to keep
    keep_indices = set(range(len(sentences)))

    # Compare each sentence with previous sentences
    for i in range(1, len(sentences)):
        if i not in keep_indices:
            continue

        for j in range(i):
            if j not in keep_indices:
                continue

            similarity = calculate_sentence_similarity(sentences[i], sentences[j])

            if similarity >= similarity_threshold:
                # Keep the shorter sentence (usually more concise)
                if len(sentences[i]) < len(sentences[j]):
                    keep_indices.discard(j)
                else:
                    keep_indices.discard(i)
                break

    # Return kept sentences in original order
    return [sentences[i] for i in sorted(keep_indices)]


def prune_redundancy(
    text: str, aggressive: bool = False, similarity_threshold: float = 0.7
) -> str:
    """Remove redundant content from text.

    Args:
        text: Input text
        aggressive: Whether to use aggressive pruning
        similarity_threshold: Threshold for semantic similarity

    Returns:
        Text with redundancy removed
    """
    # Segment into sentences
    sentences = segment_sentences(text)

    if not sentences:
        return text

    # Remove exact duplicates
    sentences = remove_duplicate_sentences(sentences)

    # Remove reformulations
    reformulation_indices = detect_reformulations(sentences)
    sentences = [s for i, s in enumerate(sentences) if i not in reformulation_indices]

    # Remove clarifying sentences
    clarifying_indices = detect_clarifying_sentences(sentences)
    sentences = [s for i, s in enumerate(sentences) if i not in clarifying_indices]

    if aggressive:
        # Remove verbose explanations
        verbose_indices = detect_verbose_explanations(sentences)
        sentences = [s for i, s in enumerate(sentences) if i not in verbose_indices]

        # Remove semantic redundancy
        sentences = remove_semantic_redundancy(sentences, similarity_threshold)

    # Rejoin sentences
    return " ".join(sentences)


class RedundancyRemover:
    """Redundancy removal pass for compression pipeline."""

    def __init__(
        self, aggressive: bool = False, similarity_threshold: float = 0.7
    ) -> None:
        """Initialize redundancy remover.

        Args:
            aggressive: Whether to use aggressive pruning
            similarity_threshold: Threshold for semantic similarity
        """
        self.aggressive = aggressive
        self.similarity_threshold = similarity_threshold

    def process(self, text: str) -> str:
        """Process text through redundancy removal.

        Args:
            text: Input text

        Returns:
            Text with redundancy removed
        """
        return prune_redundancy(text, self.aggressive, self.similarity_threshold)

    def should_run(self, state: "PipelineState") -> bool:  # type: ignore
        """Check if pass should run."""
        return True

    @property
    def name(self) -> str:
        """Get pass name."""
        return "prune"
