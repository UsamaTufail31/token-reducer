"""Base class for compression passes."""

from abc import ABC, abstractmethod

from ..types import PipelineState


class CompressionPass(ABC):
    """Abstract base class for compression passes."""

    @abstractmethod
    def process(self, text: str) -> str:
        """Process text through this compression pass.

        Args:
            text: Input text

        Returns:
            Processed text
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this pass.

        Returns:
            Pass name
        """
        pass

    def should_run(self, state: PipelineState) -> bool:
        """Determine if this pass should run given the current state.

        Args:
            state: Current pipeline state

        Returns:
            True if pass should run
        """
        # By default, always run
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
