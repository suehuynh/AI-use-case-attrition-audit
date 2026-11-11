from __future__ import annotations

import re

import pandas as pd

from .config import Config

CANONICAL_COLUMNS = [
    "uid", "year", "record_id", "agency", "bureau", "name", "name_key",
    "purpose", "outputs", "stage", "impact", "withheld",
]


def normalize_text(value: object) -> str:
    """Casefold, strip, collapse whitespace. Empty string for missing values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip().casefold()


class SchemaNormalizer:
    """Maps one year's raw columns onto the canonical schema."""

    def __init__(self, config: Config, year: str):
        self.year = year
        self.spec = config.inventory(year)["columns"]

    def _get(self, df: pd.DataFrame, key: str) -> pd.Series:
        col = self.spec.get(key)
        if col is None:
            return pd.Series([""] * len(df), index=df.index)
        if isinstance(col, list):
            parts = [df[c].fillna("").astype(str) for c in col]
            joined = parts[0]
            for p in parts[1:]:
                joined = joined.str.cat(p, sep=" ")
            return joined.str.strip()
        return df[col].fillna("").astype(str).str.strip()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["year"] = self.year
        out["record_id"] = self._get(df, "record_id")
        out["agency"] = self._get(df, "agency").str.upper()
        out["bureau"] = self._get(df, "bureau")
        out["name"] = self._get(df, "name")
        out["name_key"] = out["name"].map(normalize_text)
        out["purpose"] = self._get(df, "purpose")
        out["outputs"] = self._get(df, "outputs")
        out["stage"] = self._get(df, "stage")
        out["impact"] = self._get(df, "impact")
        out["withheld"] = self._get(df, "withheld")
        out = out.reset_index(drop=True)
        out["uid"] = [f"{self.year}-{i:05d}" for i in range(len(out))]
        return out[CANONICAL_COLUMNS]