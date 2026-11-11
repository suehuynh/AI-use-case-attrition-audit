""" Phase 1 step 1: result reproduction.
Replication of the Lurie et al. (2026) year-over-year matcher (Appendix A).

Pass 1: exact match on (use case name, agency abbreviation).
Pass 2: fuzzy match restricted to the same agency abbreviation.
        score = 0.65 * name_sim + 0.35 * purpose_sim  (threshold 65)
        name-only fallback when either purpose is missing (threshold 70)
Each record is matched at most once; ties resolved greedily by descending score.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz

from .config import Config
from .schema import normalize_text


@dataclass
class MatchResult:
    matches: pd.DataFrame   # uid_prev, uid_next, method, score
    dropped: pd.DataFrame   # prev-year records with no match
    new: pd.DataFrame       # next-year records with no match

    def summary(self) -> dict[str, int]:
        m = self.matches["method"].value_counts()
        return {
            "exact": int(m.get("exact", 0)),
            "fuzzy": int(m.get("fuzzy", 0)),
            "persisted": len(self.matches),
            "dropped": len(self.dropped),
            "new": len(self.new),
        }


class BaselineMatcher:
    def __init__(self, config: Config):
        p = config.baseline
        self.w_name = p["name_weight"]
        self.w_purpose = p["purpose_weight"]
        self.t_composite = p["composite_threshold"]
        self.t_name_only = p["name_only_threshold"]
        self.scorer = getattr(fuzz, p["scorer"])

    # ---- pass 1 -------------------------------------------------------
    @staticmethod
    def _exact(prev: pd.DataFrame, nxt: pd.DataFrame) -> list[tuple[str, str, str, float]]:
        pairs, used = [], set()
        index: dict[tuple[str, str], list[str]] = {}
        for uid, key, ag in zip(nxt["uid"], nxt["name_key"], nxt["agency"]):
            index.setdefault((key, ag), []).append(uid)
        for uid, key, ag in zip(prev["uid"], prev["name_key"], prev["agency"]):
            for cand in index.get((key, ag), []):
                if cand not in used:
                    used.add(cand)
                    pairs.append((uid, cand, "exact", 100.0))
                    break
        return pairs

    # ---- pass 2 -------------------------------------------------------
    def score(self, a: pd.Series, b: pd.Series) -> float | None:
        name_sim = self.scorer(a["name_key"], b["name_key"])
        pa, pb = normalize_text(a["purpose"]), normalize_text(b["purpose"])
        if pa and pb:
            s = self.w_name * name_sim + self.w_purpose * self.scorer(pa, pb)
            return s if s >= self.t_composite else None
        return name_sim if name_sim >= self.t_name_only else None

    def _fuzzy(self, prev: pd.DataFrame, nxt: pd.DataFrame) -> list[tuple[str, str, str, float]]:
        cands = []
        for agency, p_grp in prev.groupby("agency"):
            n_grp = nxt[nxt["agency"] == agency]
            for _, a in p_grp.iterrows():
                for _, b in n_grp.iterrows():
                    s = self.score(a, b)
                    if s is not None:
                        cands.append((s, a["uid"], b["uid"]))
        cands.sort(key=lambda x: -x[0])
        used_p, used_n, pairs = set(), set(), []
        for s, up, un in cands:
            if up in used_p or un in used_n:
                continue
            used_p.add(up)
            used_n.add(un)
            pairs.append((up, un, "fuzzy", round(s, 2)))
        return pairs

    # ---- public -------------------------------------------------------
    def match(self, prev: pd.DataFrame, nxt: pd.DataFrame) -> MatchResult:
        exact = self._exact(prev, nxt)
        done_p = {p for p, *_ in exact}
        done_n = {n for _, n, *_ in exact}
        rest_p = prev[~prev["uid"].isin(done_p)]
        rest_n = nxt[~nxt["uid"].isin(done_n)]
        fuzzy = self._fuzzy(rest_p, rest_n)

        matches = pd.DataFrame(exact + fuzzy, columns=["uid_prev", "uid_next", "method", "score"])
        dropped = prev[~prev["uid"].isin(matches["uid_prev"])].reset_index(drop=True)
        new = nxt[~nxt["uid"].isin(matches["uid_next"])].reset_index(drop=True)
        return MatchResult(matches, dropped, new)