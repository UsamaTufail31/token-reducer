"""AST-based Python code compression."""

import ast
from typing import Dict, List, Optional, Set, Tuple


class PythonASTCompressor:
    """Compresses Python code using AST transformations."""

    def __init__(
        self,
        remove_comments: bool = True,
        remove_docstrings: bool = True,
        rename_variables: bool = False,
        remove_dead_code: bool = False,
    ):
        """Initialize AST compressor.

        Args:
            remove_comments: Whether to remove comments
            remove_comments: Whether to remove docstrings
            rename_variables: Whether to rename variables to shorter names
            remove_dead_code: Whether to remove unused imports
        """
        self.remove_comments = remove_comments
        self.remove_docstrings = remove_docstrings
        self.rename_variables = rename_variables
        self.remove_dead_code = remove_dead_code

        self.rename_map: Dict[str, str] = {}

    def compress(self, code: str) -> Tuple[str, Dict[str, str]]:
        """Compress Python code.

        Args:
            code: Python code to compress

        Returns:
            Tuple of (compressed code, rename mapping)
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)

            # Apply transformations
            if self.remove_docstrings:
                tree = DocstringRemover().visit(tree)

            if self.remove_dead_code:
                tree = DeadCodeEliminator().visit(tree)

            if self.rename_variables:
                renamer = VariableRenamer()
                tree = renamer.visit(tree)
                self.rename_map = renamer.rename_map

            # Convert back to code
            compressed = ast.unparse(tree)

            # Remove comments (AST doesn't preserve comments, so they're already gone)
            # But we can clean up extra whitespace
            if self.remove_comments:
                compressed = self._clean_whitespace(compressed)

            return compressed, self.rename_map

        except SyntaxError as e:
            # If parsing fails, return original code
            return code, {}

    def _clean_whitespace(self, code: str) -> str:
        """Clean up extra whitespace.

        Args:
            code: Code to clean

        Returns:
            Cleaned code
        """
        import re

        # Remove blank lines
        lines = code.split("\n")
        non_blank_lines = [line for line in lines if line.strip()]

        return "\n".join(non_blank_lines)


class DocstringRemover(ast.NodeTransformer):
    """AST visitor that removes docstrings."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Visit function definition.

        Args:
            node: Function definition node

        Returns:
            Modified node
        """
        # Remove docstring if present
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            # This is a docstring, remove it
            node.body = node.body[1:]

        # Continue visiting child nodes
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Visit class definition.

        Args:
            node: Class definition node

        Returns:
            Modified node
        """
        # Remove docstring if present
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            node.body = node.body[1:]

        self.generic_visit(node)
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Visit module.

        Args:
            node: Module node

        Returns:
            Modified node
        """
        # Remove module docstring if present
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            node.body = node.body[1:]

        self.generic_visit(node)
        return node


class VariableRenamer(ast.NodeTransformer):
    """AST visitor that renames variables to shorter names."""

    def __init__(self):
        """Initialize variable renamer."""
        self.rename_map: Dict[str, str] = {}
        self.counter = 0
        self.scope_stack: List[Set[str]] = [set()]  # Track scopes

    def _get_short_name(self) -> str:
        """Generate a short variable name.

        Returns:
            Short name like 'a', 'b', ..., 'aa', 'ab', ...
        """
        name = ""
        n = self.counter
        while True:
            name = chr(ord("a") + (n % 26)) + name
            n = n // 26
            if n == 0:
                break
            n -= 1
        self.counter += 1
        return name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Visit function definition.

        Args:
            node: Function definition node

        Returns:
            Modified node
        """
        # Enter new scope
        self.scope_stack.append(set())

        # Rename function arguments
        for arg in node.args.args:
            if arg.arg not in self.rename_map:
                short_name = self._get_short_name()
                self.rename_map[arg.arg] = short_name
                self.scope_stack[-1].add(arg.arg)
            arg.arg = self.rename_map[arg.arg]

        # Visit body
        self.generic_visit(node)

        # Exit scope
        self.scope_stack.pop()

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Visit name node.

        Args:
            node: Name node

        Returns:
            Modified node
        """
        # Only rename local variables (not built-ins or globals)
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]

        return node


class DeadCodeEliminator(ast.NodeTransformer):
    """AST visitor that removes unused imports."""

    def __init__(self):
        """Initialize dead code eliminator."""
        self.imported_names: Set[str] = set()
        self.used_names: Set[str] = set()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Visit module.

        Args:
            node: Module node

        Returns:
            Modified node
        """
        # First pass: collect imported names
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    name = alias.asname if alias.asname else alias.name
                    self.imported_names.add(name)
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    name = alias.asname if alias.asname else alias.name
                    self.imported_names.add(name)

        # Second pass: collect used names
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                self.used_names.add(child.id)

        # Third pass: remove unused imports
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                # Check if any imported name is used
                keep = False
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        name = alias.asname if alias.asname else alias.name
                        if name in self.used_names:
                            keep = True
                            break
                elif isinstance(stmt, ast.ImportFrom):
                    for alias in stmt.names:
                        name = alias.asname if alias.asname else alias.name
                        if name in self.used_names:
                            keep = True
                            break

                if keep:
                    new_body.append(stmt)
            else:
                new_body.append(stmt)

        node.body = new_body
        return node


class CommentRemover:
    """Removes comments from Python code (non-AST based)."""

    @staticmethod
    def remove_comments(code: str) -> str:
        """Remove comments from code.

        Args:
            code: Code with comments

        Returns:
            Code without comments
        """
        import re

        # Remove single-line comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

        # Remove blank lines
        lines = code.split("\n")
        non_blank_lines = [line for line in lines if line.strip()]

        return "\n".join(non_blank_lines)
