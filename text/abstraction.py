"""Entity abstraction for semantic compression."""

import re
from typing import Dict, List, Optional, Tuple


class EntityAbstractor:
    """Abstracts named entities to reduce token count."""

    def __init__(self, use_spacy: bool = False):
        """Initialize entity abstractor.

        Args:
            use_spacy: Whether to use spacy for NER (requires spacy installed)
        """
        self.use_spacy = use_spacy
        self._spacy_nlp = None

        if use_spacy:
            try:
                import spacy

                self._spacy_nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                # Fallback to regex-based approach
                self.use_spacy = False

        # Entity mapping for reversibility
        self.entity_map: Dict[str, str] = {}
        self.entity_counters: Dict[str, int] = {
            "ORG": 0,
            "PERSON": 0,
            "LOC": 0,
            "DATE": 0,
            "MONEY": 0,
        }

    def abstract_entities(self, text: str, preserve: bool = True) -> Tuple[str, Dict[str, str]]:
        """Abstract named entities in text.

        Args:
            text: Text to process
            preserve: If True, return mapping for reversibility

        Returns:
            Tuple of (abstracted text, entity mapping)
        """
        if self.use_spacy and self._spacy_nlp:
            return self._abstract_with_spacy(text, preserve)
        else:
            return self._abstract_with_regex(text, preserve)

    def _abstract_with_spacy(self, text: str, preserve: bool) -> Tuple[str, Dict[str, str]]:
        """Abstract entities using spacy NER.

        Args:
            text: Text to process
            preserve: If True, return mapping

        Returns:
            Tuple of (abstracted text, entity mapping)
        """
        doc = self._spacy_nlp(text)
        entity_map = {}

        # Process entities in reverse order to maintain positions
        replacements = []
        for ent in doc.ents:
            entity_type = ent.label_
            if entity_type in self.entity_counters:
                self.entity_counters[entity_type] += 1
                placeholder = f"[{entity_type}{self.entity_counters[entity_type]}]"
                replacements.append((ent.start_char, ent.end_char, placeholder, ent.text))

                if preserve:
                    entity_map[placeholder] = ent.text

        # Apply replacements in reverse order
        result = text
        for start, end, placeholder, original in reversed(replacements):
            result = result[:start] + placeholder + result[end:]

        return result, entity_map

    def _abstract_with_regex(self, text: str, preserve: bool) -> Tuple[str, Dict[str, str]]:
        """Abstract entities using regex patterns (fallback).

        Args:
            text: Text to process
            preserve: If True, return mapping

        Returns:
            Tuple of (abstracted text, entity mapping)
        """
        entity_map = {}
        result = text

        # Organization patterns (simple heuristic)
        org_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Corporation|Corp|Inc|LLC|Ltd|Company|Co))\b"
        for match in re.finditer(org_pattern, text):
            self.entity_counters["ORG"] += 1
            placeholder = f"[ORG{self.entity_counters['ORG']}]"
            if preserve:
                entity_map[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)

        # Person names (capitalized words, simple heuristic)
        person_pattern = r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b"
        for match in re.finditer(person_pattern, result):
            # Skip if already replaced
            if match.group(0).startswith("["):
                continue
            self.entity_counters["PERSON"] += 1
            placeholder = f"[PERSON{self.entity_counters['PERSON']}]"
            if preserve:
                entity_map[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)

        # Locations (common city/country names - simplified)
        location_pattern = r"\b(New York|Los Angeles|London|Paris|Tokyo|Beijing|Mumbai|Islamabad|Karachi|Lahore)\b"
        for match in re.finditer(location_pattern, result, re.IGNORECASE):
            self.entity_counters["LOC"] += 1
            placeholder = f"[LOC{self.entity_counters['LOC']}]"
            if preserve:
                entity_map[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)

        # Dates
        date_pattern = r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
        for match in re.finditer(date_pattern, result):
            self.entity_counters["DATE"] += 1
            placeholder = f"[DATE{self.entity_counters['DATE']}]"
            if preserve:
                entity_map[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)

        # Money amounts
        money_pattern = r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?"
        for match in re.finditer(money_pattern, result):
            self.entity_counters["MONEY"] += 1
            placeholder = f"[MONEY{self.entity_counters['MONEY']}]"
            if preserve:
                entity_map[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)

        return result, entity_map

    def restore_entities(self, text: str, entity_map: Dict[str, str]) -> str:
        """Restore original entities from abstracted text.

        Args:
            text: Abstracted text
            entity_map: Mapping of placeholders to original entities

        Returns:
            Text with restored entities
        """
        result = text
        for placeholder, original in entity_map.items():
            result = result.replace(placeholder, original)
        return result

    def reset(self):
        """Reset entity counters and mapping."""
        self.entity_map = {}
        self.entity_counters = {
            "ORG": 0,
            "PERSON": 0,
            "LOC": 0,
            "DATE": 0,
            "MONEY": 0,
        }
