"""Compression level configuration with quality thresholds."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..types import CompressionLevel


@dataclass
class LevelConfig:
    """Configuration for a compression level."""

    level: CompressionLevel
    target_reduction_min: float  # Minimum % reduction
    target_reduction_max: float  # Maximum % reduction
    quality_threshold: float  # Minimum semantic similarity
    enable_normalization: bool = True
    enable_redundancy_removal: bool = True
    enable_semantic_compression: bool = True
    enable_summarization: bool = False
    enable_aggressive_pruning: bool = False
    preserve_comments: bool = True  # For code
    preserve_docstrings: bool = True  # For code
    preserve_verbose_explanations: bool = True  # For text
    max_processing_time_ms: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "target_reduction_min": self.target_reduction_min,
            "target_reduction_max": self.target_reduction_max,
            "quality_threshold": self.quality_threshold,
            "enable_normalization": self.enable_normalization,
            "enable_redundancy_removal": self.enable_redundancy_removal,
            "enable_semantic_compression": self.enable_semantic_compression,
            "enable_summarization": self.enable_summarization,
            "enable_aggressive_pruning": self.enable_aggressive_pruning,
            "preserve_comments": self.preserve_comments,
            "preserve_docstrings": self.preserve_docstrings,
            "preserve_verbose_explanations": self.preserve_verbose_explanations,
            "max_processing_time_ms": self.max_processing_time_ms,
            "description": self.description,
        }


# Predefined level configurations
LEVEL_CONFIGS: Dict[CompressionLevel, LevelConfig] = {
    CompressionLevel.LIGHT: LevelConfig(
        level=CompressionLevel.LIGHT,
        target_reduction_min=5.0,
        target_reduction_max=15.0,
        quality_threshold=0.98,
        enable_normalization=True,
        enable_redundancy_removal=True,
        enable_semantic_compression=False,
        enable_summarization=False,
        enable_aggressive_pruning=False,
        preserve_comments=True,
        preserve_docstrings=True,
        preserve_verbose_explanations=True,
        max_processing_time_ms=50.0,
        description="Minimal compression with maximum safety (5-15% reduction, >98% similarity)",
    ),
    CompressionLevel.MODERATE: LevelConfig(
        level=CompressionLevel.MODERATE,
        target_reduction_min=20.0,
        target_reduction_max=40.0,
        quality_threshold=0.90,
        enable_normalization=True,
        enable_redundancy_removal=True,
        enable_semantic_compression=True,
        enable_summarization=False,
        enable_aggressive_pruning=False,
        preserve_comments=False,
        preserve_docstrings=True,
        preserve_verbose_explanations=False,
        max_processing_time_ms=100.0,
        description="Balanced compression and quality (20-40% reduction, >90% similarity)",
    ),
    CompressionLevel.AGGRESSIVE: LevelConfig(
        level=CompressionLevel.AGGRESSIVE,
        target_reduction_min=50.0,
        target_reduction_max=70.0,
        quality_threshold=0.80,
        enable_normalization=True,
        enable_redundancy_removal=True,
        enable_semantic_compression=True,
        enable_summarization=True,
        enable_aggressive_pruning=True,
        preserve_comments=False,
        preserve_docstrings=False,
        preserve_verbose_explanations=False,
        max_processing_time_ms=200.0,
        description="Maximum compression with acceptable loss (50-70% reduction, >80% similarity)",
    ),
}


def get_level_config(level: CompressionLevel) -> LevelConfig:
    """Get configuration for a compression level.

    Args:
        level: Compression level

    Returns:
        LevelConfig for the level

    Raises:
        ValueError: If level is not supported
    """
    if level not in LEVEL_CONFIGS:
        raise ValueError(f"Unsupported compression level: {level}")

    return LEVEL_CONFIGS[level]


def validate_compression_level(level: CompressionLevel) -> None:
    """Validate that compression level is supported.

    Args:
        level: Compression level to validate

    Raises:
        ValueError: If level is not supported
    """
    if level not in LEVEL_CONFIGS:
        raise ValueError(
            f"Unsupported compression level: {level}. "
            f"Supported levels: {', '.join(l.value for l in LEVEL_CONFIGS.keys())}"
        )


def get_supported_levels() -> List[CompressionLevel]:
    """Get list of supported compression levels.

    Returns:
        List of supported CompressionLevel values
    """
    return list(LEVEL_CONFIGS.keys())


def get_quality_threshold(level: CompressionLevel) -> float:
    """Get quality threshold for a compression level.

    Args:
        level: Compression level

    Returns:
        Quality threshold (semantic similarity minimum)
    """
    config = get_level_config(level)
    return config.quality_threshold


def get_target_reduction(level: CompressionLevel) -> tuple[float, float]:
    """Get target reduction range for a compression level.

    Args:
        level: Compression level

    Returns:
        Tuple of (min_reduction, max_reduction) percentages
    """
    config = get_level_config(level)
    return (config.target_reduction_min, config.target_reduction_max)


class LevelValidator:
    """Validator for compression level configurations."""

    @staticmethod
    def validate_quality_threshold(threshold: float) -> None:
        """Validate quality threshold value.

        Args:
            threshold: Quality threshold to validate

        Raises:
            ValueError: If threshold is invalid
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Quality threshold must be between 0.0 and 1.0, got {threshold}"
            )

        if threshold < 0.5:
            raise ValueError(
                f"Quality threshold {threshold} is too low. "
                "Minimum recommended threshold is 0.5"
            )

    @staticmethod
    def validate_custom_config(config: LevelConfig) -> None:
        """Validate a custom level configuration.

        Args:
            config: Level configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate reduction targets
        if config.target_reduction_min < 0 or config.target_reduction_max > 100:
            raise ValueError(
                "Reduction targets must be between 0 and 100 percent"
            )

        if config.target_reduction_min > config.target_reduction_max:
            raise ValueError(
                "Minimum reduction target cannot exceed maximum reduction target"
            )

        # Validate quality threshold
        LevelValidator.validate_quality_threshold(config.quality_threshold)

        # Validate processing time
        if (
            config.max_processing_time_ms is not None
            and config.max_processing_time_ms <= 0
        ):
            raise ValueError("Max processing time must be positive")

    @staticmethod
    def check_compatibility(
        level: CompressionLevel, task_name: str
    ) -> Optional[str]:
        """Check if level is appropriate for task.

        Args:
            level: Compression level
            task_name: Task name

        Returns:
            Warning message if combination is suboptimal, None otherwise
        """
        # Aggressive compression for translation
        if task_name == "translation" and level != CompressionLevel.LIGHT:
            return (
                "Translation tasks should use LIGHT compression to avoid "
                "information loss. Consider using CompressionLevel.LIGHT."
            )

        # Light compression for extraction/RAG
        if task_name in {"extraction", "rag"} and level == CompressionLevel.LIGHT:
            return (
                f"{task_name.upper()} tasks can benefit from more aggressive "
                "compression. Consider using CompressionLevel.MODERATE or AGGRESSIVE."
            )

        return None


def create_custom_level(
    target_reduction_min: float = 10.0,
    target_reduction_max: float = 30.0,
    quality_threshold: float = 0.90,
    **kwargs,
) -> LevelConfig:
    """Create a custom compression level configuration.

    Args:
        target_reduction_min: Minimum reduction percentage
        target_reduction_max: Maximum reduction percentage
        quality_threshold: Minimum semantic similarity
        **kwargs: Additional configuration options

    Returns:
        Custom LevelConfig

    Raises:
        ValueError: If configuration is invalid
    """
    config = LevelConfig(
        level=CompressionLevel.MODERATE,  # Placeholder
        target_reduction_min=target_reduction_min,
        target_reduction_max=target_reduction_max,
        quality_threshold=quality_threshold,
        **kwargs,
    )

    LevelValidator.validate_custom_config(config)

    return config
