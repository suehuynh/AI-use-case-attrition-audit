"""Validation estimates: gold classes, design-weighted totals, accuracy and agreement.

Population: the dropped records. The pilot records are a census with their own human labels; the
validation sample (sampling.py) is drawn from the other records only, so each sampled record
carries weight N_h / n_h. Because the weights depend only on the sampling design, the same human
labels can score any model run.

Estimand: the number of dropped records whose system still appears in the 2025 individual inventory
at the SAME agency (gold class 'same_agency'). Corrected estimate:
    T = sum_h N_h * ybar_h  +  T_pilot  +  |COTS|
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

MODEL_COARSE = {"renamed": "same_agency", "consolidated": "same_agency", "split": "same_agency",
                "withheld": "same_agency", "cross_agency_move": "cross_agency",
                "moved_to_cots": "cots", "absent": "absent"}
HUMAN_MATCH = "SAME_SYSTEM"   # coders choose SAME_SYSTEM or NO_MATCH


def gold_class(relation: str, match_ids: str, found_uid: str, record_agency: str,
               key: pd.DataFrame, nxt_agency: pd.Series) -> str:
    """Map one coder's record-level answer to same_agency / cross_agency / absent."""
    relation = str(relation).strip()
    if relation != HUMAN_MATCH:
        return "absent"
    uids = set(key.loc[key["cand_id"].isin(
        {c.upper() for c in re.split(r"[,\s;]+", str(match_ids)) if c}), "uid_next"])
    if str(found_uid).strip():
        uids.add(str(found_uid).strip())
    agencies = {nxt_agency.get(u, "") for u in uids}
    if not agencies or agencies == {""}:
        return "absent"
    return "same_agency" if record_agency in agencies else "cross_agency"


def stratified_total(df: pd.DataFrame, y: str) -> tuple[float, float]:
    """Total of binary column y over the sampled strata, with its standard error
    (stratified simple random sampling with finite population correction)."""
    total, var = 0.0, 0.0
    for _, g in df.groupby("stratum"):
        N, n = g["N_h"].iloc[0], len(g)
        p = g[y].mean()
        total += N * p
        if n > 1:
            s2 = g[y].var(ddof=1)
            var += N ** 2 * (1 - n / N) * s2 / n
    return total, math.sqrt(var)


def weighted_accuracy(df: pd.DataFrame, pred: str, gold: str) -> float:
    w = df["weight"]
    return float((w * (df[pred] == df[gold])).sum() / w.sum())


def weighted_precision_recall(df: pd.DataFrame, pred: str, gold: str, cls: str) -> tuple[float, float]:
    w = df["weight"]
    tp = (w * ((df[pred] == cls) & (df[gold] == cls))).sum()
    pp = (w * (df[pred] == cls)).sum()
    gp = (w * (df[gold] == cls)).sum()
    return (float(tp / pp) if pp else float("nan"), float(tp / gp) if gp else float("nan"))


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))
