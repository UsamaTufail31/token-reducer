# Code compression module

from .ast_compressor import (
    CommentRemover,
    DeadCodeEliminator,
    DocstringRemover,
    PythonASTCompressor,
    VariableRenamer,
)

__all__ = [
    "PythonASTCompressor",
    "DocstringRemover",
    "VariableRenamer",
    "DeadCodeEliminator",
    "CommentRemover",
]
