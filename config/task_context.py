"""Task context configuration for context-aware compression."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ..types import TaskContext


@dataclass
class TaskConfig:
    """Configuration for a specific task type."""

    task: TaskContext
    preserve_entities: bool = True
    preserve_numbers: bool = True
    preserve_facts: bool = True
    preserve_causal_links: bool = False
    preserve_chronology: bool = False
    preserve_instructions: bool = True
    aggressive_pruning: bool = False
    description: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "task": self.task.value,
            "preserve_entities": self.preserve_entities,
            "preserve_numbers": self.preserve_numbers,
            "preserve_facts": self.preserve_facts,
            "preserve_causal_links": self.preserve_causal_links,
            "preserve_chronology": self.preserve_chronology,
            "preserve_instructions": self.preserve_instructions,
            "aggressive_pruning": self.aggressive_pruning,
            "description": self.description,
        }


# Predefined task configurations
TASK_CONFIGS: Dict[TaskContext, TaskConfig] = {
    TaskContext.SUMMARIZATION: TaskConfig(
        task=TaskContext.SUMMARIZATION,
        preserve_entities=True,
        preserve_numbers=True,
        preserve_facts=True,
        preserve_causal_links=True,
        preserve_chronology=True,
        preserve_instructions=False,
        aggressive_pruning=False,
        description="Preserve causal links and chronological order for summarization",
    ),
    TaskContext.RAG: TaskConfig(
        task=TaskContext.RAG,
        preserve_entities=True,
        preserve_numbers=True,
        preserve_facts=True,
        preserve_causal_links=False,
        preserve_chronology=False,
        preserve_instructions=False,
        aggressive_pruning=True,
        description="Optimize for retrieval context with high information density",
    ),
    TaskContext.EXTRACTION: TaskConfig(
        task=TaskContext.EXTRACTION,
        preserve_entities=True,
        preserve_numbers=True,
        preserve_facts=True,
        preserve_causal_links=False,
        preserve_chronology=False,
        preserve_instructions=False,
        aggressive_pruning=True,
        description="Keep only fields relevant to extraction target",
    ),
    TaskContext.REASONING: TaskConfig(
        task=TaskContext.REASONING,
        preserve_entities=True,
        preserve_numbers=True,
        preserve_facts=True,
        preserve_causal_links=True,
        preserve_chronology=False,
        preserve_instructions=True,
        aggressive_pruning=False,
        description="Preserve premises, key details, and logical connections",
    ),
    TaskContext.TRANSLATION: TaskConfig(
        task=TaskContext.TRANSLATION,
        preserve_entities=True,
        preserve_numbers=True,
        preserve_facts=True,
        preserve_causal_links=True,
        preserve_chronology=True,
        preserve_instructions=True,
        aggressive_pruning=False,
        description="Skip compression entirely for translation tasks",
    ),
    TaskContext.CODE_COMPLETION: TaskConfig(
        task=TaskContext.CODE_COMPLETION,
        preserve_entities=True,
        preserve_numbers=True,
        preserve_facts=True,
        preserve_causal_links=False,
        preserve_chronology=False,
        preserve_instructions=True,
        aggressive_pruning=False,
        description="Preserve function signatures and interfaces",
    ),
    TaskContext.DEBUGGING: TaskConfig(
        task=TaskContext.DEBUGGING,
        preserve_entities=True,
        preserve_numbers=True,
        preserve_facts=True,
        preserve_causal_links=True,
        preserve_chronology=False,
        preserve_instructions=True,
        aggressive_pruning=False,
        description="Maintain variable names and error-relevant context",
    ),
    TaskContext.QUESTION_ANSWERING: TaskConfig(
        task=TaskContext.QUESTION_ANSWERING,
        preserve_entities=True,
        preserve_numbers=True,
        preserve_facts=True,
        preserve_causal_links=False,
        preserve_chronology=False,
        preserve_instructions=False,
        aggressive_pruning=True,
        description="Preserve facts and entities for answering questions",
    ),
}


def get_task_config(task: TaskContext) -> TaskConfig:
    """Get configuration for a task type.

    Args:
        task: Task context

    Returns:
        TaskConfig for the task

    Raises:
        ValueError: If task is not supported
    """
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unsupported task: {task}")

    return TASK_CONFIGS[task]


def validate_task_context(task: TaskContext) -> None:
    """Validate that task context is supported.

    Args:
        task: Task context to validate

    Raises:
        ValueError: If task is not supported
    """
    if task not in TASK_CONFIGS:
        raise ValueError(
            f"Unsupported task: {task}. "
            f"Supported tasks: {', '.join(t.value for t in TASK_CONFIGS.keys())}"
        )


def get_supported_tasks() -> List[TaskContext]:
    """Get list of supported task types.

    Returns:
        List of supported TaskContext values
    """
    return list(TASK_CONFIGS.keys())


def should_skip_compression(task: TaskContext) -> bool:
    """Check if compression should be skipped for this task.

    Args:
        task: Task context

    Returns:
        True if compression should be skipped
    """
    # Translation tasks should skip compression
    return task == TaskContext.TRANSLATION


class TaskContextValidator:
    """Validator for task context configurations."""

    @staticmethod
    def validate(task: TaskContext, content_type: str) -> None:
        """Validate task is appropriate for content type.

        Args:
            task: Task context
            content_type: Type of content ('text' or 'code')

        Raises:
            ValueError: If task/content combination is invalid
        """
        validate_task_context(task)

        # Code-specific tasks
        code_tasks = {TaskContext.CODE_COMPLETION, TaskContext.DEBUGGING}

        if content_type == "text" and task in code_tasks:
            raise ValueError(
                f"Task {task.value} is only applicable to code, not text"
            )

        # Text-specific tasks
        text_tasks = {
            TaskContext.SUMMARIZATION,
            TaskContext.TRANSLATION,
            TaskContext.QUESTION_ANSWERING,
        }

        if content_type == "code" and task in text_tasks:
            raise ValueError(
                f"Task {task.value} is only applicable to text, not code"
            )

    @staticmethod
    def get_recommended_level(task: TaskContext) -> str:
        """Get recommended compression level for task.

        Args:
            task: Task context

        Returns:
            Recommended level ('light', 'moderate', or 'aggressive')
        """
        # Conservative tasks
        if task in {
            TaskContext.TRANSLATION,
            TaskContext.DEBUGGING,
            TaskContext.REASONING,
        }:
            return "light"

        # Aggressive tasks
        if task in {TaskContext.EXTRACTION, TaskContext.RAG}:
            return "aggressive"

        # Balanced tasks
        return "moderate"
