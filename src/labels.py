"""Swing-task labels from source outcome descriptions (config/outcome_mapping.yaml)."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["miss", "contact", "bunt", "take", "non_competitive"]


def load_mapping(source="statcast"):
    cfg = yaml.safe_load((ROOT / "config" / "outcome_mapping.yaml").read_text())
    mapping = {}
    for category in CATEGORIES:
        for value in cfg[source][category]:
            if value in mapping:
                raise ValueError(f"{value!r} mapped to both {mapping[value]} and {category}")
            mapping[value] = category
    return mapping


def outcome_category(description: pd.Series, mapping=None) -> pd.Series:
    """Map each description to a category. Missing descriptions are 'unknown'; unmapped values raise."""
    mapping = mapping or load_mapping()
    present = set(description.dropna().unique())
    unmapped = present - set(mapping)
    if unmapped:
        raise ValueError(f"unmapped outcome descriptions: {sorted(unmapped)} - review config/outcome_mapping.yaml")
    return description.map(mapping).fillna("unknown")


def whiff_label(category: pd.Series) -> pd.Series:
    """1 = miss, 0 = contact, <NA> = not an ordinary swing (take, bunt, unknown, non-competitive)."""
    return category.map({"miss": 1, "contact": 0}).astype("Int8")
