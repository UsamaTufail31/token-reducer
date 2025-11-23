"""Main public API for token_reducer library."""

from typing import List, Optional, Union

from .config import (
    CompressionLevel,
    TaskContext,
    create_tokenizer,
    get_level_config,
    get_task_config,
)
from .pipeline.orchestrator import create_text_pipeline
from .types import CompressionResult
from .utils.token_counter import calculate_metrics


def compress_text(
    text: str,
    task: Union[TaskContext, str] = TaskContext.SUMMARIZATION,
    level: Union[CompressionLevel, str] = CompressionLevel.MODERATE,
    tokenizer: str = "gpt-4",
    **kwargs,
) -> CompressionResult:
    """Compress text for LLM applications.

    Args:
        text: Text to compress
        task: Task context (TaskContext enum or string)
        level: Compression level (CompressionLevel enum or string)
        tokenizer: Tokenizer name (e.g., 'gpt-4', 'claude-3', 'gpt-3.5-turbo')
        **kwargs: Additional configuration options

    Returns:
        CompressionResult with compressed text and metrics

    Examples:
        >>> result = compress_text(
        ...     "Long article text...",
        ...     task=TaskContext.SUMMARIZATION,
        ...     level=CompressionLevel.MODERATE
        ... )
        >>> print(f"Reduced from {result.original_tokens} to {result.compressed_tokens} tokens")
        >>> print(result.compressed_text)
    """
    # Convert string inputs to enums if needed
    if isinstance(task, str):
        task = TaskContext(task.lower())

    if isinstance(level, str):
        level = CompressionLevel(level.lower())

    # Create and execute pipeline
    pipeline = create_text_pipeline(task=task, level=level, tokenizer_name=tokenizer)

    compressed_text, metrics, warnings = pipeline.execute(text)

    # Create result object
    result = CompressionResult(
        compressed_text=compressed_text,
        original_text=text,
        metrics=metrics,
        warnings=warnings,
        metadata={
            "task": task.value,
            "level": level.value,
            "tokenizer": tokenizer,
        },
    )

    return result


def compress_code(
    code: str,
    task: Union[TaskContext, str] = TaskContext.CODE_COMPLETION,
    level: Union[CompressionLevel, str] = CompressionLevel.MODERATE,
    language: str = "python",
    tokenizer: str = "gpt-4",
    **kwargs,
) -> "CodeCompressionResult":
    """Compress code for LLM applications.

    Args:
        code: Code to compress
        task: Task context (TaskContext enum or string)
        level: Compression level (CompressionLevel enum or string)
        language: Programming language
        tokenizer: Tokenizer name
        **kwargs: Additional configuration options

    Returns:
        CodeCompressionResult with compressed code and metrics

    Examples:
        >>> result = compress_code(
        ...     code="def long_function_name():\\n    ...",
        ...     task=TaskContext.CODE_COMPLETION,
        ...     level=CompressionLevel.AGGRESSIVE
        ... )
        >>> print(result.compressed_code)
    """
    from .types import CodeCompressionResult

    # Convert string inputs to enums if needed
    if isinstance(task, str):
        task = TaskContext(task.lower())

    if isinstance(level, str):
        level = CompressionLevel(level.lower())

    # For now, return a basic implementation
    # Full code compression will be implemented in code module
    tokenizer_obj = create_tokenizer(tokenizer)

    # Basic code compression: remove comments and blank lines
    import re

    compressed = code

    # Remove single-line comments
    compressed = re.sub(r"#.*$", "", compressed, flags=re.MULTILINE)

    # Remove blank lines
    compressed = "\n".join(line for line in compressed.split("\n") if line.strip())

    # Calculate metrics
    metrics = calculate_metrics(
        original_text=code,
        compressed_text=compressed,
        tokenizer=tokenizer_obj,
        processing_time_ms=0.0,
    )

    result = CodeCompressionResult(
        compressed_code=compressed,
        original_code=code,
        metrics=metrics,
        rename_mapping={},
        warnings=[],
        metadata={
            "task": task.value,
            "level": level.value,
            "language": language,
            "tokenizer": tokenizer,
        },
    )

    return result


def batch_compress_text(
    texts: List[str],
    task: Union[TaskContext, str] = TaskContext.SUMMARIZATION,
    level: Union[CompressionLevel, str] = CompressionLevel.MODERATE,
    tokenizer: str = "gpt-4",
    parallel: bool = False,
    **kwargs,
) -> List[CompressionResult]:
    """Compress multiple texts.

    Args:
        texts: List of texts to compress
        task: Task context
        level: Compression level
        tokenizer: Tokenizer name
        parallel: Whether to process in parallel (not yet implemented)
        **kwargs: Additional configuration options

    Returns:
        List of CompressionResult objects

    Examples:
        >>> results = batch_compress_text(
        ...     texts=["text1", "text2", "text3"],
        ...     task=TaskContext.RAG,
        ...     level=CompressionLevel.MODERATE
        ... )
        >>> total_reduction = sum(r.reduction_percentage for r in results) / len(results)
    """
    results = []

    for text in texts:
        result = compress_text(text=text, task=task, level=level, tokenizer=tokenizer, **kwargs)
        results.append(result)

    return results


class CompressionConfig:
    """Configuration object for compression."""

    def __init__(
        self,
        task: Union[TaskContext, str] = TaskContext.SUMMARIZATION,
        level: Union[CompressionLevel, str] = CompressionLevel.MODERATE,
        tokenizer: str = "gpt-4",
        preserve_entities: bool = True,
        preserve_numbers: bool = True,
        preserve_facts: bool = True,
        preserve_instructions: bool = True,
        quality_threshold: float = 0.90,
        enable_fail_safe: bool = True,
        # Advanced features
        enable_ast_parsing: bool = False,
        enable_semantic_dedup: bool = False,
        enable_entity_abstraction: bool = False,
        enable_proposition_extraction: bool = False,
        use_embeddings: bool = False,
        semantic_threshold: float = 0.85,
        target_tokens: Optional[int] = None,
        reversible: bool = False,
        **kwargs,
    ) -> None:
        """Initialize compression configuration.

        Args:
            task: Task context
            level: Compression level
            tokenizer: Tokenizer name
            preserve_entities: Whether to preserve named entities
            preserve_numbers: Whether to preserve numerical data
            preserve_facts: Whether to preserve factual statements
            preserve_instructions: Whether to preserve instructions
            quality_threshold: Minimum semantic similarity threshold
            enable_fail_safe: Whether to enable fail-safe mode
            enable_ast_parsing: Whether to use AST-based code compression
            enable_semantic_dedup: Whether to use semantic deduplication
            enable_entity_abstraction: Whether to abstract named entities
            enable_proposition_extraction: Whether to extract propositions
            use_embeddings: Whether to use embeddings for semantic analysis
            semantic_threshold: Threshold for semantic similarity (0.0-1.0)
            target_tokens: Target token count (for progressive compression)
            reversible: Whether to generate reversibility mappings
            **kwargs: Additional configuration options
        """
        # Convert string inputs to enums if needed
        if isinstance(task, str):
            task = TaskContext(task.lower())

        if isinstance(level, str):
            level = CompressionLevel(level.lower())

        self.task = task
        self.level = level
        self.tokenizer = tokenizer
        self.preserve_entities = preserve_entities
        self.preserve_numbers = preserve_numbers
        self.preserve_facts = preserve_facts
        self.preserve_instructions = preserve_instructions
        self.quality_threshold = quality_threshold
        self.enable_fail_safe = enable_fail_safe

        # Advanced features
        self.enable_ast_parsing = enable_ast_parsing
        self.enable_semantic_dedup = enable_semantic_dedup
        self.enable_entity_abstraction = enable_entity_abstraction
        self.enable_proposition_extraction = enable_proposition_extraction
        self.use_embeddings = use_embeddings
        self.semantic_threshold = semantic_threshold
        self.target_tokens = target_tokens
        self.reversible = reversible

        self.extra_config = kwargs

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "task": self.task.value,
            "level": self.level.value,
            "tokenizer": self.tokenizer,
            "preserve_entities": self.preserve_entities,
            "preserve_numbers": self.preserve_numbers,
            "preserve_facts": self.preserve_facts,
            "preserve_instructions": self.preserve_instructions,
            "quality_threshold": self.quality_threshold,
            "enable_fail_safe": self.enable_fail_safe,
            # Advanced features
            "enable_ast_parsing": self.enable_ast_parsing,
            "enable_semantic_dedup": self.enable_semantic_dedup,
            "enable_entity_abstraction": self.enable_entity_abstraction,
            "enable_proposition_extraction": self.enable_proposition_extraction,
            "use_embeddings": self.use_embeddings,
            "semantic_threshold": self.semantic_threshold,
            "target_tokens": self.target_tokens,
            "reversible": self.reversible,
            **self.extra_config,
        }


# Convenience function using config object
def compress_with_config(text: str, config: CompressionConfig) -> CompressionResult:
    """Compress text using a configuration object.

    Args:
        text: Text to compress
        config: CompressionConfig object

    Returns:
        CompressionResult

    Examples:
        >>> config = CompressionConfig(
        ...     task=TaskContext.RAG,
        ...     level=CompressionLevel.AGGRESSIVE,
        ...     quality_threshold=0.85
        ... )
        >>> result = compress_with_config(text, config)
    """
    return compress_text(
        text=text,
        task=config.task,
        level=config.level,
        tokenizer=config.tokenizer,
    )
