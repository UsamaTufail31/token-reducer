"""Shared type definitions for token_reducer library."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskContext(Enum):
    """Task types for context-aware compression."""

    SUMMARIZATION = "summarization"
    RAG = "rag"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    CODE_COMPLETION = "code_completion"
    DEBUGGING = "debugging"
    QUESTION_ANSWERING = "question_answering"


class CompressionLevel(Enum):
    """Compression aggressiveness levels."""

    LIGHT = "light"  # 5-15% reduction, >98% similarity
    MODERATE = "moderate"  # 20-40% reduction, >90% similarity
    AGGRESSIVE = "aggressive"  # 50-70% reduction, >80% similarity


@dataclass
class CompressionMetrics:
    """Metrics for compression results."""

    original_tokens: int
    compressed_tokens: int
    reduction_percentage: float
    processing_time_ms: float
    semantic_similarity: Optional[float] = None
    pass_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (compressed / original)."""
        if self.original_tokens == 0:
            return 0.0
        return self.compressed_tokens / self.original_tokens


@dataclass
class CompressionResult:
    """Result of text compression operation."""

    compressed_text: str
    original_text: str
    metrics: CompressionMetrics
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def original_tokens(self) -> int:
        """Get original token count."""
        return self.metrics.original_tokens

    @property
    def compressed_tokens(self) -> int:
        """Get compressed token count."""
        return self.metrics.compressed_tokens

    @property
    def reduction_percentage(self) -> float:
        """Get reduction percentage."""
        return self.metrics.reduction_percentage


@dataclass
class CodeCompressionResult:
    """Result of code compression operation."""

    compressed_code: str
    original_code: str
    metrics: CompressionMetrics
    rename_mapping: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def original_tokens(self) -> int:
        """Get original token count."""
        return self.metrics.original_tokens

    @property
    def compressed_tokens(self) -> int:
        """Get compressed token count."""
        return self.metrics.compressed_tokens

    @property
    def reduction_percentage(self) -> float:
        """Get reduction percentage."""
        return self.metrics.reduction_percentage


@dataclass
class PipelineState:
    """State object passed between compression passes."""

    content: str
    original_content: str
    task: TaskContext
    level: CompressionLevel
    tokenizer_name: str
    current_tokens: int
    original_tokens: int
    pass_history: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def add_pass(self, pass_name: str, tokens_before: int, tokens_after: int) -> None:
        """Record a compression pass execution."""
        self.pass_history.append(pass_name)
        self.metrics[pass_name] = {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "reduction": tokens_before - tokens_after,
            "reduction_pct": (
                ((tokens_before - tokens_after) / tokens_before * 100)
                if tokens_before > 0
                else 0.0
            ),
        }
        self.current_tokens = tokens_after

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


class ContentType(Enum):
    """Type of content being compressed."""

    TEXT = "text"
    CODE = "code"


class ProgrammingLanguage(Enum):
    """Supported programming languages for code compression."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    CSHARP = "csharp"
    UNKNOWN = "unknown"
