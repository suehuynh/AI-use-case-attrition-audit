"""Scoring against human labels. All uids come from the labeling sheets themselves.

PilotScorer       one model's pilot run vs coder1's pilot sheet (development set)
choose_model      the model choice rule written in the README, applied to the pilot scores
ValidationScorer  inter-coder agreement, gold classes, the corrected estimate and model accuracy
                  on the validation sample (test set)
"""
from __future__ import annotations

import pandas as pd

from .adjudicator import MATCH
from .classifier import CONF_ORDER
from .labeling import LabeledSheet
from .validation import (MODEL_COARSE, cohen_kappa, gold_class, stratified_total,
                         weighted_accuracy, weighted_precision_recall)

GOLD_CLASSES = ["same_agency", "cross_agency", "absent"]


# ---------------------------------------------------------------------------
class PilotScorer:
    """Pair level: each (record, candidate) pair is a match for coder1 if the candidate is listed.
    Record level: the record is correct if both find no match, or the model's best match
    (highest confidence, then best retrieval rank) is one of coder1's matches.
    Only labeled records are scored."""

    def __init__(self, sheet: LabeledSheet):
        self.sheet = sheet
        self.records = sheet.labeled
        c = sheet.candidates[sheet.candidates["record_uid"].isin(self.records)].copy()
        c["human_match"] = [u in sheet.matched_uids(r) for r, u in zip(c["record_uid"], c["uid_next"])]
        self.human = c.rename(columns={"record_uid": "uid_prev"})

    def _join(self, adj: pd.DataFrame) -> pd.DataFrame:
        sheet_pairs = set(zip(self.sheet.candidates["record_uid"], self.sheet.candidates["uid_next"]))
        run_pairs = set(zip(adj["uid_prev"].astype(str), adj["uid_next"].astype(str)))
        if sheet_pairs != run_pairs:
            raise ValueError(f"the run did not judge exactly the sheet's pairs "
                             f"({len(sheet_pairs - run_pairs)} missing, {len(run_pairs - sheet_pairs)} extra); "
                             "rerun it with the pilot pairs file")
        a = adj.astype({"uid_prev": str, "uid_next": str})
        df = self.human.merge(a[["uid_prev", "uid_next", "rank", "relation", "confidence",
                                 "rationale"]], on=["uid_prev", "uid_next"], how="left")
        df["model_match"] = df["relation"] == MATCH
        return df

    def record_correct(self, df: pd.DataFrame) -> pd.Series:
        out = {}
        for uid in self.records:
            human = self.sheet.matched_uids(uid)
            m = df[(df["uid_prev"] == uid) & df["model_match"]].copy()
            if m.empty:
                out[uid] = not human
            else:
                m["_c"] = m["confidence"].map(CONF_ORDER).fillna(3)
                out[uid] = m.sort_values(["_c", "rank"]).iloc[0]["uid_next"] in human
        return pd.Series(out)

    def baseline_correct(self) -> int:
        """Records the always-NO_MATCH baseline gets right."""
        return sum(not self.sheet.matched_uids(u) for u in self.records)

    def score(self, adj: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        df = self._join(adj)
        h, p = df["human_match"], df["model_match"]
        tp, fp, fn = int((h & p).sum()), int((~h & p).sum()), int((h & ~p).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        correct = self.record_correct(df)
        n = len(self.records)
        s = {
            "records": n, "pairs": len(df),
            "invalid_pairs": int(df["relation"].isna().sum()),
            "human_match_pairs": int(h.sum()), "model_match_pairs": int(p.sum()),
            "pair_precision": round(prec, 3), "pair_recall": round(rec, 3),
            "pair_f1": round(2 * prec * rec / (prec + rec), 3) if tp else 0.0,
            "pair_kappa": round(cohen_kappa(h, p), 3),
            "records_correct": int(correct.sum()),
            "record_accuracy": round(correct.sum() / n, 3),
            "baseline_records_correct": self.baseline_correct(),
            "baseline_accuracy": round(self.baseline_correct() / n, 3),
        }
        s["beats_baseline"] = s["records_correct"] > s["baseline_records_correct"]
        errors = df[h != p].sort_values(["uid_prev", "rank"])
        return s, errors


def choose_model(scores: pd.DataFrame) -> dict:
    """README rule: highest record accuracy among models that beat the always-NO_MATCH baseline;
    pair kappa breaks ties. scores: one row per model, in config order."""
    eligible = scores[scores["beats_baseline"]]
    rule = ("highest record accuracy among models that beat the always-NO_MATCH baseline; "
            "pair kappa breaks ties")
    if eligible.empty:
        return {"chosen": None, "rule": rule, "reason": "no model beats the baseline",
                "eligible": []}
    top = eligible[eligible["records_correct"] == eligible["records_correct"].max()]
    reason = "highest record accuracy"
    if len(top) > 1:
        top = top[top["pair_kappa"] == top["pair_kappa"].max()]
        reason = ("kappa tie-break" if len(top) == 1
                  else "exact tie on accuracy and kappa: first in config order")
    return {"chosen": top.iloc[0]["model"], "rule": rule, "reason": reason,
            "eligible": eligible["model"].tolist()}


# ---------------------------------------------------------------------------
class ValidationScorer:
    def __init__(self, a: LabeledSheet, b: LabeledSheet, manifest: pd.DataFrame,
                 nxt_agency: pd.Series):
        self.a, self.b = a, b
        self.manifest = manifest.set_index("uid")
        self.nxt_agency = nxt_agency

    def _gold(self, sheet: LabeledSheet, uid: str) -> str:
        return gold_class(sheet.matched_uids(uid), self.manifest.loc[uid, "agency"], self.nxt_agency)

    def compare(self) -> pd.DataFrame:
        both = [u for u in self.manifest.index if u in self.a.labeled and u in self.b.labeled]
        g = pd.DataFrame(index=pd.Index(both, name="uid"))
        g["rel_a"] = [self.a.records.loc[u, "relation"] for u in both]
        g["rel_b"] = [self.b.records.loc[u, "relation"] for u in both]
        g["gold_a"] = [self._gold(self.a, u) for u in both]
        g["gold_b"] = [self._gold(self.b, u) for u in both]
        g["ids_a"] = [self.a.matched_cand_ids(u) for u in both]
        g["ids_b"] = [self.b.matched_cand_ids(u) for u in both]
        g["found_a"] = [self.a.records.loc[u, "found_2025_uid"] for u in both]
        g["found_b"] = [self.b.records.loc[u, "found_2025_uid"] for u in both]
        g["disagree"] = (g["gold_a"] != g["gold_b"]) | (
            (g["gold_a"] != "absent") & ((g["ids_a"] != g["ids_b"]) | (g["found_a"] != g["found_b"])))
        return g

    @staticmethod
    def agreement(g: pd.DataFrame) -> dict:
        return {"records_coded_by_both": len(g),
                "kappa_relation": round(cohen_kappa(g["rel_a"], g["rel_b"]), 3),
                "kappa_gold_class": round(cohen_kappa(g["gold_a"], g["gold_b"]), 3),
                "raw_agreement_gold_class": round(float((g["gold_a"] == g["gold_b"]).mean()), 3)}

    def gold(self, g: pd.DataFrame, resolved: pd.DataFrame) -> tuple[pd.Series, int]:
        """Agreed records take the shared class; disagreements take the resolved answer.
        resolved: uid, resolved_relation, resolved_match_cand_ids, resolved_found_2025_uid."""
        res = resolved.set_index("uid") if len(resolved) else resolved
        gold, unresolved = {}, 0
        for u, r in g.iterrows():
            if not r["disagree"]:
                gold[u] = r["gold_a"]
            elif len(res) and u in res.index and str(res.loc[u, "resolved_relation"]).strip():
                x = res.loc[u]
                uids = self.a.uids_for(u, x["resolved_relation"], x["resolved_match_cand_ids"],
                                       x["resolved_found_2025_uid"])
                gold[u] = gold_class(uids, self.manifest.loc[u, "agency"], self.nxt_agency)
            else:
                unresolved += 1
        return pd.Series(gold, dtype=str), unresolved

    def estimate(self, gold: pd.Series, pilot_gold: pd.Series, n_cots: int, n_total: int) -> dict:
        """T = sum_h N_h * ybar_h + T_pilot + |COTS| (same agency)."""
        df = self.manifest.loc[gold.index].copy()
        df["gold"] = gold
        df["y_same"] = (df["gold"] == "same_agency").astype(int)
        df["y_cross"] = (df["gold"] == "cross_agency").astype(int)
        t_same, se_same = stratified_total(df, "y_same")
        t_cross, _ = stratified_total(df, "y_cross")
        t_pilot = int((pilot_gold == "same_agency").sum())
        total = t_same + t_pilot + n_cots
        return {
            "population": n_total,
            "still_in_record_same_agency_or_cots": round(total, 1),
            "share": round(total / n_total, 3),
            "ci95_share": [round((total - 1.96 * se_same) / n_total, 3),
                           round((total + 1.96 * se_same) / n_total, 3)],
            "cross_agency_estimate": round(t_cross + int((pilot_gold == "cross_agency").sum()), 1),
            "components": {"sampled_strata": round(t_same, 1), "pilot_census": t_pilot,
                           "cots_rule_based": n_cots},
            "note": "SE covers sampling error in the validation strata only; pilot labels are a "
                    "single-coder census.",
        }

    def model_scores(self, gold: pd.Series, labels: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        df = self.manifest.loc[gold.index].copy()
        df["gold"] = gold
        df["pred"] = labels.set_index("uid")["final_label"].map(MODEL_COARSE).reindex(df.index).values
        df = df[df["pred"] != "cots"]
        s = {"weighted_accuracy": round(weighted_accuracy(df, "pred", "gold"), 3)}
        for c in GOLD_CLASSES:
            p, r = weighted_precision_recall(df, "pred", "gold", c)
            s[f"{c}_precision"], s[f"{c}_recall"] = round(p, 3), round(r, 3)
        s["raw_model_count_same_agency_or_cots"] = int(
            labels["final_label"].map(MODEL_COARSE).isin(["same_agency", "cots"]).sum())
        conf = pd.crosstab(df["pred"], df["gold"]).reindex(index=GOLD_CLASSES, columns=GOLD_CLASSES,
                                                           fill_value=0)
        return s, conf


def chosen_full_run(model_choice_path) -> str:
    """Run name of the full run for the model in model_choice.json (written by score_pilot.py)."""
    import json
    from pathlib import Path

    p = Path(model_choice_path)
    if not p.exists():
        raise SystemExit(f"{p} not found: score the pilot first (scripts/score_pilot.py).")
    chosen = json.loads(p.read_text())["chosen"]
    if not chosen:
        raise SystemExit(f"{p}: no model was chosen (none beat the baseline).")
    return f"full__{chosen}"
