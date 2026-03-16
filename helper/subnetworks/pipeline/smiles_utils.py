#!/usr/bin/env python3
"""Shared SMILES normalization and safe-label utilities for the pipeline."""

from __future__ import annotations

import re
from typing import List, Optional

from rdkit import Chem


_NULL_TEXT = {"", "nan", "none", "null", "na", "<na>"}


def clean_text(value) -> str:
    """Return stripped text, or empty string for null-like values."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in _NULL_TEXT:
        return ""
    return text


def split_smiles(smiles) -> List[str]:
    """Split dot-delimited SMILES into normalized text parts."""
    text = clean_text(smiles)
    if not text:
        return []
    return [part.strip() for part in text.split(".") if part.strip()]


def normalize_smiles(smiles) -> Optional[str]:
    """Normalize one SMILES by stripping stereochemistry and canonicalizing."""
    smi = clean_text(smiles)
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def normalize_smiles_text(smiles) -> Optional[str]:
    """Normalize a possibly multi-component SMILES state string."""
    parts = []
    for part in split_smiles(smiles):
        normalized = normalize_smiles(part)
        if normalized:
            parts.append(normalized)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return ".".join(sorted(parts))


def file_safe_label(text, max_len: int | None = None) -> str:
    """Return a filesystem-safe label with conservative allowed characters."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return value[:max_len] if (max_len is not None and max_len >= 0) else value
