"""Configuration module for token_reducer."""

from .compression_level import (
    CompressionLevel,
    LevelConfig,
    LevelValidator,
    create_custom_level,
    get_level_config,
    get_quality_threshold,
    get_supported_levels,
    get_target_reduction,
    validate_compression_level,
)
from .task_context import (
    TaskConfig,
    TaskContext,
    TaskContextValidator,
    get_supported_tasks,
    get_task_config,
    should_skip_compression,
    validate_task_context,
)
from .tokenizer_config import (
    AnthropicTokenizer,
    CharacterTokenizer,
    HuggingFaceTokenizer,
    TikTokenTokenizer,
    TokenizerInterface,
    create_tokenizer,
)

__all__ = [
    # Enums
    "CompressionLevel",
    "TaskContext",
    # Configs
    "LevelConfig",
    "TaskConfig",
    # Tokenizers
    "TokenizerInterface",
    "TikTokenTokenizer",
    "HuggingFaceTokenizer",
    "AnthropicTokenizer",
    "CharacterTokenizer",
    # Factories
    "create_tokenizer",
    "create_custom_level",
    # Getters
    "get_level_config",
    "get_task_config",
    "get_quality_threshold",
    "get_target_reduction",
    "get_supported_levels",
    "get_supported_tasks",
    # Validators
    "LevelValidator",
    "TaskContextValidator",
    "validate_compression_level",
    "validate_task_context",
    # Utilities
    "should_skip_compression",
]

