from __future__ import annotations

import logging

import pandas as pd

from .config import Config
from .schema import SchemaNormalizer

log = logging.getLogger(__name__)


class InventoryLoader:
    """Reads a raw inventory file, normalizes it, and applies config filters."""

    def __init__(self, config: Config):
        self.config = config

    def load_raw(self, year: str) -> pd.DataFrame:
        spec = self.config.inventory(year)
        path = self.config.raw_dir / spec["file"]
        return pd.read_csv(path, encoding=spec["encoding"], low_memory=False, dtype=str)

    def load(self, year: str) -> pd.DataFrame:
        df = SchemaNormalizer(self.config, year).transform(self.load_raw(year))
        return self.apply_filters(df, year)

    def apply_filters(self, df: pd.DataFrame, year: str) -> pd.DataFrame:
        f = self.config.filters
        n0 = len(df)
        if f.get("drop_missing_name"):
            df = df[df["name_key"] != ""]
        if f.get("dedupe_name_agency"):
            df = df.drop_duplicates(["name_key", "agency"], keep="first")
        if f.get("drop_withheld") and year == "2025":
            df = df[~df["withheld"].str.lower().str.startswith("yes")]
        log.info("%s: %d raw -> %d after filters", year, n0, len(df))
        return df.reset_index(drop=True)