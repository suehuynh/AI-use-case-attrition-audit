"""Phase 1 step 4: turn pairwise LLM verdicts into one final label per dropped record.

Rules (docs/codebook.md, "Final record labels"), applied to each record's best match
(highest confidence, then best retrieval rank):

  no match            -> moved_to_cots if the record matches a same-agency 2025 COTS product,
                         else absent
  2+ different same-agency entries matched (codebook v3: a coder lists every piece of a split;
      the model calls SAME_SYSTEM on each piece), or a SPLIT_INTO relation (exploratory runs) -> split
  best 2025 record is on the withheld list              -> withheld
  any match at a different agency                       -> cross_agency_move
      (checked before consolidation: merging across agencies is implausible, and these
       pairs are usually the same product at two agencies; Phase 2 checks them)
  PART_OF, same agency (exploratory runs only)          -> consolidated
  SAME_SYSTEM, same agency, 2025 record is shared       -> consolidated
      (shared = also the best match of another SAME-AGENCY dropped record, or already
       matched to a persisting 2024 record by the baseline)
  SAME_SYSTEM, same agency                              -> renamed
"""
from __future__ import annotations

import pandas as pd

MATCH = {"SAME_SYSTEM", "PART_OF", "SPLIT_INTO"}
CONF_ORDER = {"high": 0, "medium": 1, "low": 2}
LABEL_ORDER = ["renamed", "cross_agency_move", "consolidated", "split", "withheld",
               "moved_to_cots", "absent"]


class AttritionClassifier:
    def __init__(self, nxt: pd.DataFrame, baseline_targets: set[str], cots_uids: set[str],
                 split_from_multiple_matches: bool = True):
        self.nxt = nxt.set_index("uid")
        # True for codebook v3 (two relations). False reproduces the exploratory five-relation runs,
        # where several SAME_SYSTEM candidates meant the model was unsure, not that the entry split.
        self.split_from_multiple = split_from_multiple_matches
        self.baseline_targets = baseline_targets
        self.cots_uids = cots_uids

    @staticmethod
    def best_matches(adj: pd.DataFrame, min_confidence: str = "low") -> pd.DataFrame:
        """One row per dropped record: its best match pair (records with no match are absent)."""
        allowed = [c for c, v in CONF_ORDER.items() if v <= CONF_ORDER[min_confidence]]
        m = adj[adj["relation"].isin(MATCH) & adj["confidence"].isin(allowed)].copy()
        m["_c"] = m["confidence"].map(CONF_ORDER)
        m = m.sort_values(["uid_prev", "_c", "rank"])
        n_split = m[m["relation"] == "SPLIT_INTO"].groupby("uid_prev").size().rename("n_split")
        best = m.groupby("uid_prev", as_index=False).first().drop(columns="_c")
        return best.merge(n_split, on="uid_prev", how="left").fillna({"n_split": 0})

    def classify(self, dropped: pd.DataFrame, adj: pd.DataFrame,
                 min_confidence: str = "low") -> pd.DataFrame:
        best = self.best_matches(adj, min_confidence)
        # 2025 targets claimed by more than one dropped record of the SAME agency
        best["_prev_agency"] = best["uid_prev"].map(dropped.set_index("uid")["agency"])
        best["_next_agency"] = best["uid_next"].map(self.nxt["agency"])
        same = best[best["_prev_agency"] == best["_next_agency"]]
        dup_targets = set(same.loc[same.duplicated("uid_next", keep=False), "uid_next"])
        best = best.set_index("uid_prev")

        # Number of DIFFERENT same-agency 2025 entries matched per record (pieces of a split)
        allowed = [c for c, v in CONF_ORDER.items() if v <= CONF_ORDER[min_confidence]]
        m = adj[adj["relation"].isin(MATCH) & adj["confidence"].isin(allowed)].copy()
        m["_pa"] = m["uid_prev"].map(dropped.set_index("uid")["agency"])
        m["_na"] = m["uid_next"].map(self.nxt["agency"])
        pieces = m[m["_pa"] == m["_na"]].groupby("uid_prev")["uid_next"].nunique()

        rows = []
        for r in dropped.itertuples(index=False):
            out = {"uid": r.uid, "agency": r.agency, "name": r.name,
                   "prior_explanation": getattr(r, "prior_explanation", ""),
                   "matched_uid_next": "", "matched_agency": "", "matched_name": "",
                   "relation": "", "confidence": "", "consolidation_reason": ""}
            if r.uid not in best.index:
                out["final_label"] = "moved_to_cots" if r.uid in self.cots_uids else "absent"
                rows.append(out)
                continue
            b = best.loc[r.uid]
            tgt = self.nxt.loc[b["uid_next"]]
            out.update(matched_uid_next=b["uid_next"], matched_agency=tgt["agency"],
                       matched_name=tgt["name"], relation=b["relation"],
                       confidence=b["confidence"])
            shared = b["uid_next"] in dup_targets or b["uid_next"] in self.baseline_targets
            multi = self.split_from_multiple and pieces.get(r.uid, 0) >= 2
            if b["relation"] == "SPLIT_INTO" or b["n_split"] >= 2 or multi:
                label = "split"
            elif str(tgt.get("withheld", "")).lower().startswith("yes"):
                label = "withheld"
            elif tgt["agency"] != r.agency:
                label = "cross_agency_move"
            elif b["relation"] == "PART_OF" or shared:
                label = "consolidated"
                out["consolidation_reason"] = (
                    "part_of" if b["relation"] == "PART_OF"
                    else "shared_with_dropped" if b["uid_next"] in dup_targets
                    else "shared_with_persisting")
            else:
                label = "renamed"
            out["final_label"] = label
            rows.append(out)
        return pd.DataFrame(rows)
