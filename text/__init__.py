# Text compression module

from .abstraction import EntityAbstractor
from .propositions import PropositionExtractor
from .semantics import SemanticDeduplicator
from .summarize import HierarchicalSummarizer, TaskSpecificTightener

__all__ = [
    "EntityAbstractor",
    "PropositionExtractor",
    "SemanticDeduplicator",
    "HierarchicalSummarizer",
    "TaskSpecificTightener",
]
