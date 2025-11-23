"""Pipeline state management for compression."""

from typing import Any, Dict, List, Optional

from ..types import CompressionLevel, PipelineState, TaskContext


def create_pipeline_state(
    content: str,
    task: TaskContext,
    level: CompressionLevel,
    tokenizer_name: str,
    initial_token_count: int,
) -> PipelineState:
    """Create initial pipeline state.

    Args:
        content: Content to compress
        task: Task context
        level: Compression level
        tokenizer_name: Name of tokenizer
        initial_token_count: Initial token count

    Returns:
        PipelineState object
    """
    return PipelineState(
        content=content,
        original_content=content,
        task=task,
        level=level,
        tokenizer_name=tokenizer_name,
        current_tokens=initial_token_count,
        original_tokens=initial_token_count,
        pass_history=[],
        metrics={},
        metadata={},
        warnings=[],
    )


class StateManager:
    """Manages pipeline state across compression passes."""

    def __init__(self, state: PipelineState) -> None:
        """Initialize state manager.

        Args:
            state: Initial pipeline state
        """
        self._state = state
        self._snapshots: List[PipelineState] = []

    @property
    def state(self) -> PipelineState:
        """Get current state."""
        return self._state

    def update_content(self, new_content: str, token_count: int) -> None:
        """Update content and token count.

        Args:
            new_content: New content
            token_count: New token count
        """
        self._state.content = new_content
        self._state.current_tokens = token_count

    def record_pass(
        self, pass_name: str, tokens_before: int, tokens_after: int
    ) -> None:
        """Record a pass execution.

        Args:
            pass_name: Name of the pass
            tokens_before: Token count before pass
            tokens_after: Token count after pass
        """
        self._state.add_pass(pass_name, tokens_before, tokens_after)

    def add_warning(self, warning: str) -> None:
        """Add a warning.

        Args:
            warning: Warning message
        """
        self._state.add_warning(warning)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self._state.metadata[key] = value

    def snapshot(self) -> None:
        """Create a snapshot of current state."""
        import copy

        self._snapshots.append(copy.deepcopy(self._state))

    def restore_last_snapshot(self) -> bool:
        """Restore last snapshot.

        Returns:
            True if snapshot was restored, False if no snapshots
        """
        if not self._snapshots:
            return False

        self._state = self._snapshots.pop()
        return True

    def get_reduction_percentage(self) -> float:
        """Calculate current reduction percentage.

        Returns:
            Reduction percentage
        """
        if self._state.original_tokens == 0:
            return 0.0

        reduction = (
            self._state.original_tokens - self._state.current_tokens
        ) / self._state.original_tokens
        return reduction * 100

    def has_reached_target(self, target_min: float, target_max: float) -> bool:
        """Check if target reduction has been reached.

        Args:
            target_min: Minimum target reduction percentage
            target_max: Maximum target reduction percentage

        Returns:
            True if target reached
        """
        current_reduction = self.get_reduction_percentage()
        return target_min <= current_reduction <= target_max

    def get_pass_summary(self) -> Dict[str, Any]:
        """Get summary of all passes.

        Returns:
            Dictionary with pass summary
        """
        return {
            "passes_executed": len(self._state.pass_history),
            "pass_names": self._state.pass_history,
            "total_reduction": self.get_reduction_percentage(),
            "metrics": self._state.metrics,
        }
