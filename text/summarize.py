"""Task-specific text tightening for compression."""

from typing import List

from ..types import TaskContext
from .normalize import segment_sentences


def tighten_for_summarization(text: str) -> str:
    """Optimize text for summarization tasks.

    Preserves causal links and chronological order.

    Args:
        text: Input text

    Returns:
        Tightened text
    """
    # For summarization, preserve chronological markers
    # Already handled by semantic compression preserving facts and entities
    return text


def tighten_for_extraction(text: str) -> str:
    """Optimize text for extraction tasks.

    Keeps only relevant fields, removes narrative.

    Args:
        text: Input text

    Returns:
        Tightened text
    """
    sentences = segment_sentences(text)

    # For extraction, keep only sentences with structured data
    # (entities, numbers, facts)
    # This is aggressive - remove narrative sentences
    import re

    structured_sentences = []
    for sentence in sentences:
        # Keep if contains: numbers, dates, names (capitals), or list markers
        has_structure = bool(
            re.search(r"\d+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|^[-*•]\s", sentence)
        )
        if has_structure:
            structured_sentences.append(sentence)

    return " ".join(structured_sentences) if structured_sentences else text


def tighten_for_reasoning(text: str) -> str:
    """Optimize text for reasoning tasks.

    Preserves premises and logical connections.

    Args:
        text: Input text

    Returns:
        Tightened text
    """
    # Preserve logical connectors
    # Already handled by semantic compression
    return text


def tighten_for_rag(text: str) -> str:
    """Optimize text for RAG tasks.

    Maximizes information density for retrieval.

    Args:
        text: Input text

    Returns:
        Tightened text
    """
    # For RAG, convert to dense bullet-point style
    sentences = segment_sentences(text)

    # Remove transitional phrases
    import re

    transitions = [
        r"^(however|moreover|furthermore|additionally|also|besides),?\s*",
        r"^(therefore|thus|hence|consequently),?\s*",
        r"^(meanwhile|in addition|on the other hand),?\s*",
    ]

    cleaned_sentences = []
    for sentence in sentences:
        for pattern in transitions:
            sentence = re.sub(pattern, "", sentence, flags=re.IGNORECASE)
        if sentence.strip():
            cleaned_sentences.append(sentence.strip())

    return " ".join(cleaned_sentences)


def apply_task_specific_tightening(text: str, task: TaskContext) -> str:
    """Apply task-specific tightening strategies.

    Args:
        text: Input text
        task: Task context

    Returns:
        Task-optimized text
    """
    if task == TaskContext.TRANSLATION:
        # Skip compression for translation
        return text

    if task == TaskContext.SUMMARIZATION:
        return tighten_for_summarization(text)

    if task == TaskContext.EXTRACTION:
        return tighten_for_extraction(text)

    if task == TaskContext.REASONING:
        return tighten_for_reasoning(text)

    if task == TaskContext.RAG:
        return tighten_for_rag(text)

    if task == TaskContext.QUESTION_ANSWERING:
        # Similar to RAG - maximize information density
        return tighten_for_rag(text)

    # Default: no additional tightening
    return text


class TaskSpecificTightener:
    """Task-specific tightening pass for compression pipeline."""

    def __init__(self, task: TaskContext) -> None:
        """Initialize tightener.

        Args:
            task: Task context
        """
        self.task = task

    def process(self, text: str) -> str:
        """Process text through task-specific tightening.

        Args:
            text: Input text

        Returns:
            Task-optimized text
        """
        return apply_task_specific_tightening(text, self.task)

    def should_run(self, state: "PipelineState") -> bool:  # type: ignore
        """Check if pass should run."""
        return True

    @property
    def name(self) -> str:
        """Get pass name."""
        return "summarize"


class HierarchicalSummarizer:
    """Hierarchical summarization at different levels (sentence, paragraph, document)."""

    def __init__(self):
        """Initialize hierarchical summarizer."""
        pass

    def summarize_sentence(self, sentence: str) -> str:
        """Summarize a single sentence by removing non-essential words.

        Args:
            sentence: Sentence to summarize

        Returns:
            Summarized sentence
        """
        import re

        # Remove adverbs and adjectives (simple heuristic)
        # Keep nouns, verbs, and essential words
        words = sentence.split()

        # Common words to remove
        remove_words = {
            "very",
            "really",
            "quite",
            "rather",
            "extremely",
            "incredibly",
            "absolutely",
            "completely",
            "totally",
            "entirely",
        }

        filtered_words = [w for w in words if w.lower() not in remove_words]

        return " ".join(filtered_words)

    def summarize_paragraph(self, paragraph: str, max_sentences: int = 3) -> str:
        """Summarize a paragraph to a maximum number of sentences.

        Args:
            paragraph: Paragraph to summarize
            max_sentences: Maximum number of sentences to keep

        Returns:
            Summarized paragraph
        """
        from .normalize import segment_sentences

        sentences = segment_sentences(paragraph)

        if len(sentences) <= max_sentences:
            return paragraph

        # Score sentences by importance (simple heuristic: length and position)
        scored_sentences = []
        for i, sent in enumerate(sentences):
            # First and last sentences are often important
            position_score = 1.0 if i == 0 or i == len(sentences) - 1 else 0.5

            # Longer sentences often contain more information
            length_score = min(1.0, len(sent.split()) / 20.0)

            # Check for important keywords
            import re

            important_keywords = r"\b(important|key|main|significant|critical|essential|primary)\b"
            keyword_score = 1.0 if re.search(important_keywords, sent, re.IGNORECASE) else 0.0

            total_score = position_score + length_score + keyword_score
            scored_sentences.append((sent, total_score))

        # Sort by score and take top N
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]

        # Maintain original order
        result_sentences = []
        for sent in sentences:
            if sent in top_sentences:
                result_sentences.append(sent)

        return " ".join(result_sentences)

    def summarize_document(self, text: str, target_ratio: float = 0.5) -> str:
        """Summarize an entire document to a target ratio of original length.

        Args:
            text: Document text to summarize
            target_ratio: Target length as ratio of original (0.0-1.0)

        Returns:
            Summarized document
        """
        import re

        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if len(paragraphs) == 0:
            return text

        # Calculate target number of paragraphs
        target_paragraphs = max(1, int(len(paragraphs) * target_ratio))

        # Score paragraphs
        scored_paragraphs = []
        for i, para in enumerate(paragraphs):
            # First and last paragraphs are important
            position_score = 2.0 if i == 0 or i == len(paragraphs) - 1 else 1.0

            # Length score
            length_score = min(2.0, len(para.split()) / 50.0)

            # Keyword score
            important_keywords = r"\b(conclusion|summary|important|key|main|result|finding)\b"
            keyword_score = 2.0 if re.search(important_keywords, para, re.IGNORECASE) else 0.0

            total_score = position_score + length_score + keyword_score
            scored_paragraphs.append((para, total_score))

        # Sort by score and take top N
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
        top_paragraphs = [p[0] for p in scored_paragraphs[:target_paragraphs]]

        # Maintain original order
        result_paragraphs = []
        for para in paragraphs:
            if para in top_paragraphs:
                result_paragraphs.append(para)

        return "\n\n".join(result_paragraphs)

    def summarize(self, text: str, level: str = "document", **kwargs) -> str:
        """Summarize text at specified level.

        Args:
            text: Text to summarize
            level: Level of summarization ('sentence', 'paragraph', 'document')
            **kwargs: Additional arguments for specific levels

        Returns:
            Summarized text
        """
        if level == "sentence":
            return self.summarize_sentence(text)
        elif level == "paragraph":
            max_sentences = kwargs.get("max_sentences", 3)
            return self.summarize_paragraph(text, max_sentences)
        else:  # document
            target_ratio = kwargs.get("target_ratio", 0.5)
            return self.summarize_document(text, target_ratio)
