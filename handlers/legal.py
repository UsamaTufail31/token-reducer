"""Legal document compression handler."""

import re
from typing import List, Set, Tuple


class LegalHandler:
    """Specialized handler for legal document compression."""

    def __init__(self):
        """Initialize legal handler."""
        self.boilerplate_patterns = [
            r"This\s+(?:Agreement|Contract|Document)\s+is\s+entered\s+into.*?(?:\n|\.)",
            r"IN\s+WITNESS\s+WHEREOF.*?(?:\n|\.)",
            r"The\s+parties\s+hereto\s+have\s+executed.*?(?:\n|\.)",
            r"(?:Signed|Executed)\s+this\s+\d+.*?day\s+of.*?(?:\n|\.)",
        ]

        self.legal_terms = {
            "whereas",
            "hereby",
            "herein",
            "thereof",
            "pursuant",
            "aforementioned",
            "hereinafter",
            "notwithstanding",
        }

    def compress_legal_document(self, document: str, preserve_clauses: bool = True) -> Tuple[str, List[str]]:
        """Compress legal document.

        Args:
            document: Legal document text
            preserve_clauses: Whether to preserve all clauses

        Returns:
            Tuple of (compressed document, removed sections)
        """
        removed_sections = []

        # Remove boilerplate if not preserving all clauses
        if not preserve_clauses:
            for pattern in self.boilerplate_patterns:
                matches = re.findall(pattern, document, re.IGNORECASE | re.DOTALL)
                removed_sections.extend(matches)
                document = re.sub(pattern, "", document, flags=re.IGNORECASE | re.DOTALL)

        # Simplify legal jargon
        document = self._simplify_legal_language(document)

        # Clean up whitespace
        document = re.sub(r"\n\s*\n\s*\n+", "\n\n", document)

        return document, removed_sections

    def _simplify_legal_language(self, text: str) -> str:
        """Simplify legal language.

        Args:
            text: Legal text

        Returns:
            Simplified text
        """
        # Simplifications
        simplifications = [
            (r"pursuant\s+to", "under"),
            (r"in\s+the\s+event\s+that", "if"),
            (r"for\s+the\s+purpose\s+of", "to"),
            (r"with\s+respect\s+to", "about"),
            (r"in\s+accordance\s+with", "per"),
            (r"notwithstanding\s+the\s+foregoing", "however"),
        ]

        for pattern, replacement in simplifications:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def extract_clauses(self, document: str) -> List[Tuple[str, str]]:
        """Extract numbered clauses from document.

        Args:
            document: Legal document

        Returns:
            List of (clause_number, clause_text) tuples
        """
        # Pattern for numbered clauses
        clause_pattern = re.compile(
            r"(?:Section|Article|Clause)\s+(\d+(?:\.\d+)*)[:\s]+(.+?)(?=(?:Section|Article|Clause)\s+\d+|$)",
            re.IGNORECASE | re.DOTALL,
        )

        clauses = clause_pattern.findall(document)
        return [(num.strip(), text.strip()) for num, text in clauses]

    def extract_definitions(self, document: str) -> dict:
        """Extract defined terms from document.

        Args:
            document: Legal document

        Returns:
            Dictionary of term -> definition
        """
        definitions = {}

        # Pattern for definitions: "Term" means ...
        definition_pattern = re.compile(r'"([^"]+)"\s+means\s+(.+?)(?:\.|;|\n)', re.IGNORECASE)

        matches = definition_pattern.findall(document)
        for term, definition in matches:
            definitions[term.strip()] = definition.strip()

        return definitions

    def abstract_definitions(self, document: str) -> Tuple[str, dict]:
        """Replace defined terms with placeholders.

        Args:
            document: Legal document

        Returns:
            Tuple of (abstracted document, term mapping)
        """
        definitions = self.extract_definitions(document)

        term_map = {}
        result = document

        for i, term in enumerate(definitions.keys(), 1):
            placeholder = f"[TERM{i}]"
            term_map[placeholder] = term

            # Replace term in document (case-insensitive)
            result = re.sub(rf'\b{re.escape(term)}\b', placeholder, result, flags=re.IGNORECASE)

        return result, term_map

    def summarize_legal_document(self, document: str) -> str:
        """Create a summary of legal document.

        Args:
            document: Legal document

        Returns:
            Summary
        """
        clauses = self.extract_clauses(document)
        definitions = self.extract_definitions(document)

        summary_parts = ["Legal Document Summary\n"]

        if definitions:
            summary_parts.append(f"Defined Terms: {len(definitions)}")
            for term in list(definitions.keys())[:5]:
                summary_parts.append(f"  - {term}")
            summary_parts.append("")

        if clauses:
            summary_parts.append(f"Total Clauses: {len(clauses)}")
            summary_parts.append("\nKey Clauses:")
            for num, text in clauses[:5]:
                preview = text[:100] + "..." if len(text) > 100 else text
                summary_parts.append(f"  {num}: {preview}")

        return "\n".join(summary_parts)
