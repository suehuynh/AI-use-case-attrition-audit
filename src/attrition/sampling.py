"""Samples of dropped records for hand labeling.

PilotSampler       50 records, stratified by prior explanation (development set, coder1 only)
ValidationSampler  ~100 non-pilot records, stratified by the chosen model's final label
                   (test set, coder1 and coder2; no prompt tuning on it)

Validation strata: the model's final label, except that records retired in 2024 that still got a
match form their own stratum ('retired_matched'). moved_to_cots is rule-based (name match to the
COTS checklist), so it is not sampled and is counted as still in the record. Each sampled record
carries weight N_h / n_h; the weights depend only on the sampling design.
"""
from __future__ import annotations

import pandas as pd


class PilotSampler:
    """Pilot sample stratified by prior explanation, so explained drops are also checked."""

    @staticmethod
    def sample(dropped: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        frac = n / len(dropped)
        parts = [g.sample(max(1, round(len(g) * frac)), random_state=seed)
                 for _, g in dropped.groupby("prior_explanation")]
        return pd.concat(parts).head(n).reset_index(drop=True)


def stratum_of(row: pd.Series) -> str:
    if "retired" in str(row["prior_explanation"]) and row["final_label"] not in ("absent",
                                                                                  "moved_to_cots"):
        return "retired_matched"
    return row["final_label"]


class ValidationSampler:
    def __init__(self, allocation: dict[str, int], seed: int):
        self.allocation = allocation
        self.seed = seed

    def sample(self, labels: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
        pool = labels[~labels["uid"].isin(exclude)].copy()
        pool["stratum"] = pool.apply(stratum_of, axis=1)
        pool = pool[pool["stratum"] != "moved_to_cots"]
        parts = []
        for h, g in pool.groupby("stratum"):
            n_h = min(self.allocation.get(h, 0), len(g))
            if n_h == 0:
                raise ValueError(f"stratum '{h}' ({len(g)} records) has no allocation; "
                                 "every stratum needs at least one sampled record")
            s = g.sample(n_h, random_state=self.seed).copy()
            s["N_h"], s["n_h"] = len(g), n_h
            s["weight"] = len(g) / n_h
            parts.append(s)
        return pd.concat(parts).sort_values(["stratum", "uid"]).reset_index(drop=True)
