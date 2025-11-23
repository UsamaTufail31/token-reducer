"""Token Reducer - Intelligent token reduction for LLM applications.

Reduce token counts by 50-70% while preserving semantic meaning and context.
"""

__version__ = "0.2.0"

from .api import (
    CompressionConfig,
    batch_compress_text,
    compress_code,
    compress_text,
    compress_with_config,
)
from .types import (
    CodeCompressionResult,
    CompressionLevel,
    CompressionMetrics,
    CompressionResult,
    TaskContext,
)

# Advanced compression modules (optional imports)
try:
    from .code import PythonASTCompressor
    from .handlers import LegalHandler, LogHandler, TranscriptHandler
    from .pipeline import ContentIdentifier, ContentType, Segmenter
    from .text import (
        EntityAbstractor,
        HierarchicalSummarizer,
        PropositionExtractor,
        SemanticDeduplicator,
    )

    _ADVANCED_AVAILABLE = True
except ImportError:
    _ADVANCED_AVAILABLE = False

__all__ = [
    # Main API functions
    "compress_text",
    "compress_code",
    "batch_compress_text",
    "compress_with_config",
    # Configuration
    "CompressionConfig",
    # Enums
    "TaskContext",
    "CompressionLevel",
    # Result types
    "CompressionResult",
    "CodeCompressionResult",
    "CompressionMetrics",
    # Advanced modules (if available)
    "PythonASTCompressor",
    "EntityAbstractor",
    "PropositionExtractor",
    "SemanticDeduplicator",
    "HierarchicalSummarizer",
    "ContentIdentifier",
    "ContentType",
    "Segmenter",
    "LogHandler",
    "TranscriptHandler",
    "LegalHandler",
    # Version
    "__version__",
]

