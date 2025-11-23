"""Semantic deduplication using embeddings or heuristics."""

from typing import List, Optional, Set, Tuple


class SemanticDeduplicator:
    """Removes semantically redundant sentences."""

    def __init__(self, use_embeddings: bool = False, similarity_threshold: float = 0.85):
        """Initialize semantic deduplicator.

        Args:
            use_embeddings: Whether to use sentence embeddings (requires sentence-transformers)
            similarity_threshold: Threshold for considering sentences similar (0.0-1.0)
        """
        self.use_embeddings = use_embeddings
        self.similarity_threshold = similarity_threshold
        self._model = None

        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                # Fallback to heuristic approach
                self.use_embeddings = False

    def deduplicate(self, sentences: List[str]) -> List[str]:
        """Remove semantically duplicate sentences.

        Args:
            sentences: List of sentences to deduplicate

        Returns:
            List of unique sentences
        """
        if not sentences:
            return []

        if self.use_embeddings and self._model:
            return self._deduplicate_with_embeddings(sentences)
        else:
            return self._deduplicate_with_heuristics(sentences)

    def _deduplicate_with_embeddings(self, sentences: List[str]) -> List[str]:
        """Deduplicate using sentence embeddings.

        Args:
            sentences: List of sentences

        Returns:
            Deduplicated sentences
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Encode sentences
        embeddings = self._model.encode(sentences)

        # Track which sentences to keep
        keep_indices = []
        seen_embeddings = []

        for i, emb in enumerate(embeddings):
            is_duplicate = False

            # Check similarity with already kept sentences
            for seen_emb in seen_embeddings:
                similarity = cosine_similarity([emb], [seen_emb])[0][0]
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep_indices.append(i)
                seen_embeddings.append(emb)

        return [sentences[i] for i in keep_indices]

    def _deduplicate_with_heuristics(self, sentences: List[str]) -> List[str]:
        """Deduplicate using heuristic similarity (fallback).

        Args:
            sentences: List of sentences

        Returns:
            Deduplicated sentences
        """
        unique_sentences = []
        seen_normalized = set()

        for sentence in sentences:
            # Normalize sentence for comparison
            normalized = self._normalize_sentence(sentence)

            # Check if we've seen a similar sentence
            if normalized not in seen_normalized:
                # Check for partial matches
                is_duplicate = False
                for seen in seen_normalized:
                    if self._are_similar_heuristic(normalized, seen):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_sentences.append(sentence)
                    seen_normalized.add(normalized)

        return unique_sentences

    def _normalize_sentence(self, sentence: str) -> str:
        """Normalize sentence for comparison.

        Args:
            sentence: Sentence to normalize

        Returns:
            Normalized sentence
        """
        import re

        # Convert to lowercase
        normalized = sentence.lower()

        # Remove punctuation
        normalized = re.sub(r"[^\w\s]", "", normalized)

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def _are_similar_heuristic(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are similar using heuristics.

        Args:
            sent1: First sentence (normalized)
            sent2: Second sentence (normalized)

        Returns:
            True if sentences are similar
        """
        # Jaccard similarity of words
        words1 = set(sent1.split())
        words2 = set(sent2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        jaccard = intersection / union if union > 0 else 0

        return jaccard >= self.similarity_threshold

    def deduplicate_text(self, text: str, sentence_separator: str = ". ") -> str:
        """Deduplicate sentences in text.

        Args:
            text: Text to deduplicate
            sentence_separator: Separator to use when joining sentences

        Returns:
            Deduplicated text
        """
        import re

        # Split into sentences
        sentences = re.split(r"[.!?]+\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Deduplicate
        unique_sentences = self.deduplicate(sentences)

        # Rejoin
        return sentence_separator.join(unique_sentences)
