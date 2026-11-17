"""Inter-coder agreement, model accuracy, and the corrected attrition estimate.

Usage
  python scripts/score_validation.py                 # after both coders finish
  python scripts/score_validation.py --runs full__Qwen2.5-7B-Instruct full__gemma-2-9b-it

Steps
1. Reads both coders' sheets (uids come from the sheets) and checks them against the pairs file;
   reports agreement (kappa) on relation and on the gold class.
2. Writes paths.disagreements. Resolve each row together: fill resolved_relation,
   resolved_match_cand_ids (and resolved_found_2025_uid if needed), then rerun this script.
   Unresolved disagreements are left out of the estimates (a warning says how many).
3. Gold class per record: same_agency / cross_agency / absent.
4. Corrected estimate T = sum_h N_h * ybar_h + T_pilot + |COTS| (pilot census from coder1's
   pilot sheet; COTS = rule-based moved_to_cots records outside the pilot).
5. For each model run (default: the chosen model's full run): weighted accuracy and per-class
   precision/recall against gold.
Writes paths.validation_report and confusion__<run>.csv next to it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.labeling import LabeledSheet  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402
from attrition.scoring import ValidationScorer, chosen_full_run  # noqa: E402
from attrition.validation import gold_class  # noqa: E402

RESOLVED = ["resolved_relation", "resolved_match_cand_ids", "resolved_found_2025_uid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", nargs="*", default=None, help="model runs to score (final/<run>)")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    v = cfg.raw["validation"]
    out = cfg.output_dir
    main_run = chosen_full_run(cfg.path("model_choice"))
    runs = args.runs or [main_run]

    pairs = pd.read_csv(cfg.path("validation_pairs"), dtype=str)
    a_name, b_name = v["coders"][:2]
    A = LabeledSheet.read(cfg.path("validation_sheet", coder=a_name))
    B = LabeledSheet.read(cfg.path("validation_sheet", coder=b_name))
    A.check_pairs(pairs)
    B.check_pairs(pairs)
    manifest = pd.read_csv(cfg.path("validation_manifest"), dtype={"uid": str})
    nxt_agency = InventoryLoader(cfg).load("2025").set_index("uid")["agency"]
    scorer = ValidationScorer(A, B, manifest, nxt_agency)

    g = scorer.compare()
    if g.empty:
        sys.exit("No record has been labeled by both coders yet.")
    agreement = scorer.agreement(g)

    # Disagreements: keep earlier resolutions, add new rows
    res_path = cfg.path("disagreements")
    resolved = (pd.read_csv(res_path, dtype=str).fillna("") if res_path.exists()
                else pd.DataFrame(columns=["uid"] + RESOLVED))
    keep = resolved.set_index("uid")[RESOLVED] if len(resolved) else None
    dis = g[g["disagree"]].reset_index()
    table = pd.DataFrame({
        "uid": dis["uid"],
        f"{a_name}_relation": dis["rel_a"], f"{a_name}_ids": dis["ids_a"], f"{a_name}_found": dis["found_a"],
        f"{b_name}_relation": dis["rel_b"], f"{b_name}_ids": dis["ids_b"], f"{b_name}_found": dis["found_b"],
    })
    for c in RESOLVED:
        table[c] = [keep.loc[u, c] if keep is not None and u in keep.index else "" for u in dis["uid"]]
    table.to_csv(res_path, index=False)

    gold, unresolved = scorer.gold(g, table)
    if unresolved:
        print(f"WARNING: {unresolved} disagreements unresolved; left out of estimates. "
              f"Resolve them in {res_path} and rerun.")

    # Pilot census (coder1) and rule-based COTS records
    pilot_sheet = LabeledSheet.read(cfg.path("pilot_sheet"))
    pilot_sheet.check_pairs(pd.read_csv(cfg.path("pilot_pairs"), dtype=str))
    dropped = pd.read_csv(out / "dropped_tagged.csv", dtype=str).set_index("uid")
    pilot_gold = pd.Series({u: gold_class(pilot_sheet.matched_uids(u), dropped.loc[u, "agency"], nxt_agency)
                            for u in pilot_sheet.labeled}, dtype=str)
    labels = pd.read_csv(out / "final" / main_run / "record_labels.csv", dtype=str).fillna("")
    n_cots = int(((labels["final_label"] == "moved_to_cots") & ~labels["uid"].isin(pilot_gold.index)).sum())
    corrected = scorer.estimate(gold, pilot_gold, n_cots, len(labels))

    report_path = cfg.path("validation_report")
    models = {}
    for run in runs:
        lab = pd.read_csv(out / "final" / run / "record_labels.csv", dtype=str).fillna("")
        models[run], conf = scorer.model_scores(gold, lab)
        conf.to_csv(report_path.parent / f"confusion__{run}.csv")

    df = manifest.set_index("uid").loc[gold.index].assign(gold=gold)
    report = {"agreement": agreement, "unresolved_disagreements": unresolved,
              "corrected_estimate": corrected, "models": models,
              "gold_by_stratum": pd.crosstab(df["stratum"], df["gold"]).to_dict("index")}
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
