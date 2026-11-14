"""Phase 2: inter-coder agreement, model accuracy, and the corrected attrition estimate.

Usage
  python scripts/score_validation.py                       # after both coders finish
  python scripts/score_validation.py --runs full__Qwen2.5-7B-Instruct__adjudicator_v1 \
                                            full__Qwen2.5-7B-Instruct__adjudicator_v2

Steps
1. Reads both coders' sheets; reports agreement (kappa) on relation and on the gold class.
2. Writes outputs/validation/disagreements.csv. Resolve each row together: fill resolved_relation,
   resolved_match_cand_ids (and resolved_found_2025_uid if needed), then rerun this script.
   Until a disagreement is resolved, that record is left out of the estimates (a warning says how many).
3. Gold class per record: same_agency / cross_agency / absent.
4. Corrected estimate: stratified (design-weighted) total of same_agency over the non-pilot
   population, plus the pilot census (Sue's labels) and the rule-based moved_to_cots records.
5. For each model run: weighted accuracy and per-class precision/recall against gold.
Writes outputs/validation/validation_report.json and confusion__<run>.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.labeling import HEADER_ROW  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402
from attrition.validation import (MODEL_COARSE, cohen_kappa, gold_class,  # noqa: E402
                                  stratified_total, weighted_accuracy,
                                  weighted_precision_recall)

CLASSES = ["same_agency", "cross_agency", "absent"]


def read_sheet(path: Path) -> pd.DataFrame:
    s = pd.read_excel(path, sheet_name="labels", header=HEADER_ROW - 1, dtype=str).fillna("")
    s = s[s["role"] == "2024"].copy()
    if "found_2025_uid" not in s.columns:
        s["found_2025_uid"] = ""
    return s.set_index("record_uid")[["agency", "match_cand_ids", "relation", "confidence",
                                      "found_2025_uid", "notes"]]


def norm_ids(x: str) -> str:
    return ",".join(sorted({c.upper() for c in re.split(r"[,\s;]+", str(x)) if c}))


def pilot_census(cfg: Config, nxt_agency: pd.Series) -> pd.Series:
    """Gold class for the 50 pilot records from Sue's reviewed labels (single coder)."""
    ldir = cfg.output_dir / "labeling"
    sheet = pd.read_excel(ldir / "pilot_labeling_sheet.xlsx", header=7, dtype=str).fillna("")
    sheet = sheet[(sheet["role"] == "2024") & (sheet["relation"].str.strip() != "")]
    key = pd.read_csv(ldir / "pilot_labeling_key.csv")
    return pd.Series({r.record_uid: gold_class(r.relation, r.match_cand_ids, "", r.agency,
                                               key[key["record_uid"] == r.record_uid], nxt_agency)
                      for r in sheet.itertuples()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="model runs to score (final/<run>)")
    args = ap.parse_args()

    cfg = Config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    v = cfg.raw["validation"]
    out = cfg.output_dir
    vdir = out / "validation"
    runs = args.runs or [v["run"]]

    manifest = pd.read_csv(vdir / "sample_manifest.csv", dtype={"uid": str}).set_index("uid")
    key = pd.read_csv(vdir / "validation_key.csv")
    nxt_agency = InventoryLoader(cfg).load("2025").set_index("uid")["agency"]
    a_name, b_name = v["coders"][:2]
    A = read_sheet(vdir / f"validation_sheet__{a_name}.xlsx")
    B = read_sheet(vdir / f"validation_sheet__{b_name}.xlsx")

    def gold_for(row, uid):
        return gold_class(row["relation"], row["match_cand_ids"], row["found_2025_uid"],
                          manifest.loc[uid, "agency"], key[key["record_uid"] == uid], nxt_agency)

    both = [u for u in manifest.index
            if A.loc[u, "relation"].strip() and B.loc[u, "relation"].strip()]
    if not both:
        sys.exit("No record has been labeled by both coders yet.")
    g = pd.DataFrame(index=both)
    g["rel_a"], g["rel_b"] = A.loc[both, "relation"], B.loc[both, "relation"]
    g["gold_a"] = [gold_for(A.loc[u], u) for u in both]
    g["gold_b"] = [gold_for(B.loc[u], u) for u in both]
    g["ids_a"] = [norm_ids(A.loc[u, "match_cand_ids"]) for u in both]
    g["ids_b"] = [norm_ids(B.loc[u, "match_cand_ids"]) for u in both]

    agreement = {
        "records_coded_by_both": len(both),
        "kappa_relation": round(cohen_kappa(g["rel_a"], g["rel_b"]), 3),
        "kappa_gold_class": round(cohen_kappa(g["gold_a"], g["gold_b"]), 3),
        "raw_agreement_gold_class": round(float((g["gold_a"] == g["gold_b"]).mean()), 3),
    }

    # Disagreements on gold class or on the chosen candidate
    dis_mask = (g["gold_a"] != g["gold_b"]) | ((g["gold_a"] != "absent") & (g["ids_a"] != g["ids_b"]))
    res_path = vdir / "disagreements.csv"
    resolved = {}
    if res_path.exists():
        prev = pd.read_csv(res_path, dtype=str).fillna("")
        for r in prev.itertuples():
            if r.resolved_relation.strip():
                resolved[r.uid] = r
    dis = g[dis_mask]
    rows = []
    for u in dis.index:
        old = resolved.get(u)
        rows.append({"uid": u, f"{a_name}_relation": g.loc[u, "rel_a"],
                     f"{a_name}_ids": g.loc[u, "ids_a"], f"{b_name}_relation": g.loc[u, "rel_b"],
                     f"{b_name}_ids": g.loc[u, "ids_b"],
                     f"{a_name}_found": A.loc[u, "found_2025_uid"],
                     f"{b_name}_found": B.loc[u, "found_2025_uid"],
                     "resolved_relation": old.resolved_relation if old else "",
                     "resolved_match_cand_ids": old.resolved_match_cand_ids if old else "",
                     "resolved_found_2025_uid": old.resolved_found_2025_uid if old else ""})
    pd.DataFrame(rows).to_csv(res_path, index=False)

    gold = {}
    for u in both:
        if not dis_mask[u]:
            gold[u] = g.loc[u, "gold_a"]
        elif u in resolved:
            r = resolved[u]
            gold[u] = gold_class(r.resolved_relation, r.resolved_match_cand_ids,
                                 r.resolved_found_2025_uid, manifest.loc[u, "agency"],
                                 key[key["record_uid"] == u], nxt_agency)
    unresolved = int(dis_mask.sum()) - sum(1 for u in dis.index if u in resolved)
    if unresolved:
        print(f"WARNING: {unresolved} disagreements unresolved; left out of estimates. "
              f"Resolve them in {res_path} and rerun.")

    df = manifest.loc[list(gold)].copy()
    df["gold"] = pd.Series(gold)
    df["y_same"] = (df["gold"] == "same_agency").astype(int)
    df["y_cross"] = (df["gold"] == "cross_agency").astype(int)

    # Corrected estimate = sampled strata (weighted) + pilot census + rule-based COTS records
    t_same, se_same = stratified_total(df, "y_same")
    t_cross, se_cross = stratified_total(df, "y_cross")
    pilot = pilot_census(cfg, nxt_agency)
    labels = pd.read_csv(out / "final" / v["run"] / "record_labels.csv", dtype=str).fillna("")
    n_cots = int(((labels["final_label"] == "moved_to_cots") & ~labels["uid"].isin(pilot.index)).sum())
    n_total = len(labels)
    same_total = t_same + int((pilot == "same_agency").sum()) + n_cots
    corrected = {
        "population": n_total,
        "still_in_record_same_agency_or_cots": round(same_total, 1),
        "share": round(same_total / n_total, 3),
        "ci95_share": [round((same_total - 1.96 * se_same) / n_total, 3),
                       round((same_total + 1.96 * se_same) / n_total, 3)],
        "cross_agency_estimate": round(t_cross + int((pilot == "cross_agency").sum()), 1),
        "components": {"sampled_strata": round(t_same, 1),
                       "pilot_census": int((pilot == "same_agency").sum()),
                       "cots_rule_based": n_cots},
        "note": "SE covers sampling error in the validation strata only; pilot labels are a "
                "single-coder census.",
    }

    model_scores = {}
    for run in runs:
        lab = pd.read_csv(out / "final" / run / "record_labels.csv", dtype=str).fillna("")
        pred = lab.set_index("uid")["final_label"].map(MODEL_COARSE)
        d = df.assign(pred=pred.reindex(df.index).values)
        d = d[d["pred"] != "cots"]
        s = {"weighted_accuracy": round(weighted_accuracy(d, "pred", "gold"), 3)}
        for c in CLASSES:
            p, r = weighted_precision_recall(d, "pred", "gold", c)
            s[f"{c}_precision"], s[f"{c}_recall"] = round(p, 3), round(r, 3)
        raw = int(lab["final_label"].map(MODEL_COARSE).isin(["same_agency", "cots"]).sum())
        s["raw_model_count_same_agency_or_cots"] = raw
        model_scores[run] = s
        pd.crosstab(d["pred"], d["gold"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0) \
            .to_csv(vdir / f"confusion__{run}.csv")

    report = {"agreement": agreement, "unresolved_disagreements": unresolved,
              "corrected_estimate": corrected, "models": model_scores,
              "gold_by_stratum": pd.crosstab(df["stratum"], df["gold"]).to_dict("index")}
    (vdir / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()