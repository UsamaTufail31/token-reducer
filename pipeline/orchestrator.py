"""Multi-pass pipeline orchestrator for compression."""

import time
from typing import List, Optional

from ..config.compression_level import get_level_config
from ..config.task_context import should_skip_compression
from ..config.tokenizer_config import TokenizerInterface, create_tokenizer
from ..types import CompressionLevel, CompressionMetrics, PipelineState, TaskContext
from ..utils.token_counter import MetricsTracker
from .pass_base import CompressionPass
from .state import StateManager, create_pipeline_state


class PipelineOrchestrator:
    """Orchestrates multi-pass compression pipeline."""

    def __init__(
        self,
        task: TaskContext,
        level: CompressionLevel,
        tokenizer: TokenizerInterface,
        passes: List[CompressionPass],
        enable_early_stopping: bool = True,
    ) -> None:
        """Initialize pipeline orchestrator.

        Args:
            task: Task context
            level: Compression level
            tokenizer: Tokenizer to use
            passes: List of compression passes to execute
            enable_early_stopping: Whether to stop when target is reached
        """
        self.task = task
        self.level = level
        self.tokenizer = tokenizer
        self.passes = passes
        self.enable_early_stopping = enable_early_stopping

        self.level_config = get_level_config(level)

    def execute(self, content: str) -> tuple[str, CompressionMetrics, List[str]]:
        """Execute the compression pipeline.

        Args:
            content: Content to compress

        Returns:
            Tuple of (compressed_content, metrics, warnings)
        """
        # Check if compression should be skipped
        if should_skip_compression(self.task):
            # Return original content with minimal metrics
            metrics = CompressionMetrics(
                original_tokens=len(content.split()),
                compressed_tokens=len(content.split()),
                reduction_percentage=0.0,
                processing_time_ms=0.0,
            )
            return content, metrics, ["Compression skipped for translation task"]

        # Initialize metrics tracker
        tracker = MetricsTracker(self.tokenizer)
        tracker.start()

        # Count initial tokens
        initial_tokens = self.tokenizer.count_tokens(content)

        # Create initial state
        state = create_pipeline_state(
            content=content,
            task=self.task,
            level=self.level,
            tokenizer_name=self.tokenizer.name,
            initial_token_count=initial_tokens,
        )

        # Create state manager
        state_manager = StateManager(state)

        # Execute passes sequentially
        for pass_obj in self.passes:
            # Check if pass should run
            if not pass_obj.should_run(state):
                continue

            # Take snapshot before pass
            state_manager.snapshot()

            # Get tokens before
            tokens_before = state.current_tokens
            content_before = state.content

            try:
                # Execute pass
                content_after = pass_obj.process(state.content)

                # Count tokens after
                tokens_after = self.tokenizer.count_tokens(content_after)

                # Update state
                state_manager.update_content(content_after, tokens_after)
                state_manager.record_pass(pass_obj.name, tokens_before, tokens_after)

                # Record in tracker
                tracker.record_pass(pass_obj.name, content_before, content_after)

            except Exception as e:
                # Pass failed - restore snapshot and continue
                state_manager.restore_last_snapshot()
                state_manager.add_warning(
                    f"Pass '{pass_obj.name}' failed: {str(e)}. Skipping."
                )
                continue

            # Check for early stopping
            if self.enable_early_stopping:
                target_min, target_max = (
                    self.level_config.target_reduction_min,
                    self.level_config.target_reduction_max,
                )

                if state_manager.has_reached_target(target_min, target_max):
                    state_manager.add_metadata("early_stopped", True)
                    state_manager.add_metadata(
                        "early_stop_reason", "Target reduction reached"
                    )
                    break

        # Finalize metrics
        final_metrics = tracker.finalize(
            original_text=state.original_content,
            compressed_text=state.content,
            semantic_similarity=None,  # Will be added by fail-safe if enabled
        )

        return state.content, final_metrics, state.warnings


def create_text_pipeline(
    task: TaskContext,
    level: CompressionLevel,
    tokenizer_name: str = "gpt-4",
) -> PipelineOrchestrator:
    """Create a text compression pipeline.

    Args:
        task: Task context
        level: Compression level
        tokenizer_name: Name of tokenizer to use

    Returns:
        Configured PipelineOrchestrator
    """
    from ..text.compress import SemanticCompressor
    from ..text.normalize import TextNormalizer
    from ..text.prune import RedundancyRemover
    from ..text.repack import TextRepackager
    from ..text.summarize import TaskSpecificTightener

    # Get level configuration
    level_config = get_level_config(level)

    # Create tokenizer
    tokenizer = create_tokenizer(tokenizer_name)

    # Build pass list based on level configuration
    passes: List[CompressionPass] = []

    # Always include normalization
    if level_config.enable_normalization:
        passes.append(TextNormalizer(preserve_structure=True))

    # Add redundancy removal
    if level_config.enable_redundancy_removal:
        aggressive = level_config.enable_aggressive_pruning
        passes.append(RedundancyRemover(aggressive=aggressive))

    # Add semantic compression
    if level_config.enable_semantic_compression:
        passes.append(
            SemanticCompressor(
                preserve_entities=True,
                preserve_numbers=True,
                preserve_facts=True,
                preserve_instructions=True,
                aggressive=level_config.enable_aggressive_pruning,
            )
        )

    # Add task-specific tightening
    if level_config.enable_summarization:
        passes.append(TaskSpecificTightener(task=task))

    # Always add repackaging
    passes.append(TextRepackager(aggressive=level_config.enable_aggressive_pruning))

    return PipelineOrchestrator(
        task=task,
        level=level,
        tokenizer=tokenizer,
        passes=passes,
        enable_early_stopping=True,
    )


class PipelineBuilder:
    """Builder for creating custom compression pipelines."""

    def __init__(self) -> None:
        """Initialize pipeline builder."""
        self._task: Optional[TaskContext] = None
        self._level: Optional[CompressionLevel] = None
        self._tokenizer_name: str = "gpt-4"
        self._passes: List[CompressionPass] = []
        self._early_stopping: bool = True

    def with_task(self, task: TaskContext) -> "PipelineBuilder":
        """Set task context.

        Args:
            task: Task context

        Returns:
            Self for chaining
        """
        self._task = task
        return self

    def with_level(self, level: CompressionLevel) -> "PipelineBuilder":
        """Set compression level.

        Args:
            level: Compression level

        Returns:
            Self for chaining
        """
        self._level = level
        return self

    def with_tokenizer(self, tokenizer_name: str) -> "PipelineBuilder":
        """Set tokenizer.

        Args:
            tokenizer_name: Tokenizer name

        Returns:
            Self for chaining
        """
        self._tokenizer_name = tokenizer_name
        return self

    def add_pass(self, pass_obj: CompressionPass) -> "PipelineBuilder":
        """Add a compression pass.

        Args:
            pass_obj: Compression pass

        Returns:
            Self for chaining
        """
        self._passes.append(pass_obj)
        return self

    def with_early_stopping(self, enabled: bool) -> "PipelineBuilder":
        """Enable/disable early stopping.

        Args:
            enabled: Whether to enable early stopping

        Returns:
            Self for chaining
        """
        self._early_stopping = enabled
        return self

    def build(self) -> PipelineOrchestrator:
        """Build the pipeline.

        Returns:
            Configured PipelineOrchestrator

        Raises:
            ValueError: If required parameters are missing
        """
        if self._task is None:
            raise ValueError("Task context is required")

        if self._level is None:
            raise ValueError("Compression level is required")

        if not self._passes:
            raise ValueError("At least one compression pass is required")

        tokenizer = create_tokenizer(self._tokenizer_name)

        return PipelineOrchestrator(
            task=self._task,
            level=self._level,
            tokenizer=tokenizer,
            passes=self._passes,
            enable_early_stopping=self._early_stopping,
        )
