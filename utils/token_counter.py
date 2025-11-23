"""Token counting utilities with before/after metrics."""

import time
from typing import Optional

from ..config.tokenizer_config import TokenizerInterface, create_tokenizer
from ..types import CompressionMetrics


class TokenCounter:
    """Utility class for counting tokens and tracking metrics."""

    def __init__(self, tokenizer: TokenizerInterface) -> None:
        """Initialize token counter.

        Args:
            tokenizer: Tokenizer implementation to use
        """
        self._tokenizer = tokenizer
        self._cache: dict[str, int] = {}

    def count(self, text: str, use_cache: bool = True) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for
            use_cache: Whether to use cached results

        Returns:
            Number of tokens
        """
        if use_cache and text in self._cache:
            return self._cache[text]

        count = self._tokenizer.count_tokens(text)

        if use_cache:
            self._cache[text] = count

        return count

    def clear_cache(self) -> None:
        """Clear the token count cache."""
        self._cache.clear()

    @property
    def tokenizer_name(self) -> str:
        """Get the tokenizer name."""
        return self._tokenizer.name


def create_token_counter(tokenizer_name: str) -> TokenCounter:
    """Create a token counter with specified tokenizer.

    Args:
        tokenizer_name: Name of tokenizer to use

    Returns:
        TokenCounter instance
    """
    tokenizer = create_tokenizer(tokenizer_name)
    return TokenCounter(tokenizer)


def calculate_metrics(
    original_text: str,
    compressed_text: str,
    tokenizer: TokenizerInterface,
    processing_time_ms: float,
    semantic_similarity: Optional[float] = None,
    pass_metrics: Optional[dict] = None,
) -> CompressionMetrics:
    """Calculate compression metrics.

    Args:
        original_text: Original uncompressed text
        compressed_text: Compressed text
        tokenizer: Tokenizer to use for counting
        processing_time_ms: Processing time in milliseconds
        semantic_similarity: Optional semantic similarity score
        pass_metrics: Optional per-pass metrics

    Returns:
        CompressionMetrics object
    """
    counter = TokenCounter(tokenizer)

    original_tokens = counter.count(original_text)
    compressed_tokens = counter.count(compressed_text)

    if original_tokens == 0:
        reduction_percentage = 0.0
    else:
        reduction_percentage = (
            (original_tokens - compressed_tokens) / original_tokens * 100
        )

    return CompressionMetrics(
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        reduction_percentage=round(reduction_percentage, 2),
        processing_time_ms=round(processing_time_ms, 2),
        semantic_similarity=semantic_similarity,
        pass_metrics=pass_metrics or {},
    )


class MetricsTracker:
    """Track metrics across compression pipeline."""

    def __init__(self, tokenizer: TokenizerInterface) -> None:
        """Initialize metrics tracker.

        Args:
            tokenizer: Tokenizer to use for counting
        """
        self._counter = TokenCounter(tokenizer)
        self._start_time: Optional[float] = None
        self._pass_metrics: dict = {}

    def start(self) -> None:
        """Start timing."""
        self._start_time = time.perf_counter()

    def record_pass(
        self, pass_name: str, text_before: str, text_after: str
    ) -> None:
        """Record metrics for a compression pass.

        Args:
            pass_name: Name of the pass
            text_before: Text before the pass
            text_after: Text after the pass
        """
        tokens_before = self._counter.count(text_before)
        tokens_after = self._counter.count(text_after)

        reduction = tokens_before - tokens_after
        reduction_pct = (
            (reduction / tokens_before * 100) if tokens_before > 0 else 0.0
        )

        self._pass_metrics[pass_name] = {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "reduction": reduction,
            "reduction_pct": round(reduction_pct, 2),
        }

    def finalize(
        self,
        original_text: str,
        compressed_text: str,
        semantic_similarity: Optional[float] = None,
    ) -> CompressionMetrics:
        """Finalize and return metrics.

        Args:
            original_text: Original text
            compressed_text: Compressed text
            semantic_similarity: Optional similarity score

        Returns:
            CompressionMetrics object
        """
        if self._start_time is None:
            processing_time_ms = 0.0
        else:
            elapsed = time.perf_counter() - self._start_time
            processing_time_ms = elapsed * 1000

        return calculate_metrics(
            original_text=original_text,
            compressed_text=compressed_text,
            tokenizer=self._counter._tokenizer,
            processing_time_ms=processing_time_ms,
            semantic_similarity=semantic_similarity,
            pass_metrics=self._pass_metrics.copy(),
        )

    @property
    def pass_metrics(self) -> dict:
        """Get pass-level metrics."""
        return self._pass_metrics.copy()
