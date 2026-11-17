"""Turn pairwise SAME_SYSTEM / DIFFERENT verdicts into one final label per dropped record.

A record's matches are its SAME_SYSTEM pairs (at or above min_confidence). Its best match is the
one with the highest confidence, then the best retrieval rank. Rules, in priority order:

  split              2+ different same-agency 2025 entries matched (the codebook lists every
                     piece of a split; the model calls SAME_SYSTEM on each piece)
  withheld           the best match is on the 2025 withheld list
  cross_agency_move  the best match is at a different agency
                     (checked before consolidation: merging across agencies is implausible, and
                      these pairs are usually the same product at two agencies)
  consolidated       the best match is shared: also the best match of another SAME-AGENCY dropped
                     record, or already matched to a persisting 2024 record by the baseline
  renamed            one same-agency match, not shared
  moved_to_cots      no match, but a name match to the agency's product in the 2025 COTS checklist
  absent             no match
"""
from __future__ import annotations

import pandas as pd

from .adjudicator import MATCH

CONF_ORDER = {"high": 0, "medium": 1, "low": 2}
LABEL_ORDER = ["split", "withheld", "cross_agency_move", "consolidated", "renamed",
               "moved_to_cots", "absent"]


class AttritionClassifier:
    def __init__(self, nxt: pd.DataFrame, baseline_targets: set[str], cots_uids: set[str]):
        self.nxt = nxt.set_index("uid")
        self.baseline_targets = baseline_targets
        self.cots_uids = cots_uids

    def matches(self, dropped: pd.DataFrame, adj: pd.DataFrame,
                min_confidence: str = "low") -> pd.DataFrame:
        """All SAME_SYSTEM pairs at or above min_confidence, with both agencies attached."""
        allowed = [c for c, v in CONF_ORDER.items() if v <= CONF_ORDER[min_confidence]]
        m = adj[(adj["relation"] == MATCH) & adj["confidence"].isin(allowed)].copy()
        m["prev_agency"] = m["uid_prev"].map(dropped.set_index("uid")["agency"])
        m["next_agency"] = m["uid_next"].map(self.nxt["agency"])
        m["_c"] = m["confidence"].map(CONF_ORDER)
        return m.sort_values(["uid_prev", "_c", "rank"]).drop(columns="_c")

    def classify(self, dropped: pd.DataFrame, adj: pd.DataFrame,
                 min_confidence: str = "low") -> pd.DataFrame:
        m = self.matches(dropped, adj, min_confidence)
        same = m[m["prev_agency"] == m["next_agency"]]
        pieces = same.groupby("uid_prev")["uid_next"].nunique()
        best = m.groupby("uid_prev").first()
        # 2025 targets that are the best match of more than one dropped record of the same agency
        best_same = best[best["prev_agency"] == best["next_agency"]]
        dup_targets = set(best_same.loc[best_same["uid_next"].duplicated(keep=False), "uid_next"])

        rows = []
        for r in dropped.itertuples(index=False):
            out = {"uid": r.uid, "agency": r.agency, "name": r.name,
                   "prior_explanation": getattr(r, "prior_explanation", ""),
                   "n_matches": 0, "matched_uid_next": "", "matched_agency": "",
                   "matched_name": "", "confidence": "", "consolidation_reason": ""}
            if r.uid not in best.index:
                out["final_label"] = "moved_to_cots" if r.uid in self.cots_uids else "absent"
                rows.append(out)
                continue
            b = best.loc[r.uid]
            tgt = self.nxt.loc[b["uid_next"]]
            out.update(n_matches=int((m["uid_prev"] == r.uid).sum()),
                       matched_uid_next=b["uid_next"], matched_agency=tgt["agency"],
                       matched_name=tgt["name"], confidence=b["confidence"])
            if pieces.get(r.uid, 0) >= 2:
                label = "split"
            elif str(tgt.get("withheld", "")).lower().startswith("yes"):
                label = "withheld"
            elif tgt["agency"] != r.agency:
                label = "cross_agency_move"
            elif b["uid_next"] in dup_targets:
                label, out["consolidation_reason"] = "consolidated", "shared_with_dropped"
            elif b["uid_next"] in self.baseline_targets:
                label, out["consolidation_reason"] = "consolidated", "shared_with_persisting"
            else:
                label = "renamed"
            out["final_label"] = label
            rows.append(out)
        return pd.DataFrame(rows)
