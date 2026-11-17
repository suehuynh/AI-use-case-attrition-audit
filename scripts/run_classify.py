"""Phase 1 step 4: final label per dropped record, plus summary tables.

Usage
  python scripts/run_classify.py --run full__Qwen2.5-7B-Instruct

Reads  outputs/adjudication/<run>.csv, outputs/dropped_tagged.csv,
       outputs/baseline_matches.csv, outputs/cots_candidates.csv (optional)
Writes outputs/final/<run>/record_labels.csv   one row per dropped record
       outputs/final/<run>/by_agency.csv       label counts per 2024 agency
       outputs/final/<run>/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.classifier import LABEL_ORDER, AttritionClassifier  # noqa: E402
from attrition.config import Config  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402


def counts(s: pd.Series) -> dict:
    c = s.value_counts()
    return {k: int(c.get(k, 0)) for k in LABEL_ORDER}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", required=True, help="adjudication run name (file stem)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    out = cfg.output_dir
    adj = pd.read_csv(out / "adjudication" / f"{args.run}.csv")
    dropped = pd.read_csv(out / "dropped_tagged.csv", dtype=str).fillna("")
    nxt = InventoryLoader(cfg).load("2025")
    baseline_targets = set(pd.read_csv(out / "baseline_matches.csv", dtype=str)["uid_next"])
    cots_path = out / "cots_candidates.csv"
    cots = set(pd.read_csv(cots_path, dtype=str)["uid"]) if cots_path.exists() else set()
    if not cots_path.exists():
        print("NOTE: no cots_candidates.csv; run scripts/build_cots_candidates.py first, "
              "otherwise moved_to_cots is 0.")

    clf = AttritionClassifier(nxt, baseline_targets, cots)
    labels = clf.classify(dropped, adj)
    labels["final_label_high_conf"] = clf.classify(dropped, adj, "high")["final_label"]

    run_dir = out / "final" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(run_dir / "record_labels.csv", index=False)

    by_agency = pd.crosstab(labels["agency"], labels["final_label"]).reindex(
        columns=LABEL_ORDER, fill_value=0)
    by_agency["total"] = by_agency.sum(axis=1)
    by_agency.sort_values("total", ascending=False).to_csv(run_dir / "by_agency.csv")

    n = len(labels)
    linked = labels["final_label"].isin(
        ["renamed", "cross_agency_move", "consolidated", "split", "withheld", "moved_to_cots"])
    summary = {
        "run": args.run,
        "n_dropped": n,
        "labels": counts(labels["final_label"]),
        "labels_high_conf_only": counts(labels["final_label_high_conf"]),
        "still_in_record_share": round(linked.mean(), 3),
        "absent_share": round((labels["final_label"] == "absent").mean(), 3),
        "consolidation_reasons": labels["consolidation_reason"].replace("", pd.NA).dropna()
                                       .value_counts().to_dict(),
        "by_prior_explanation": pd.crosstab(labels["prior_explanation"], labels["final_label"])
                                  .reindex(columns=LABEL_ORDER, fill_value=0).to_dict("index"),
        "invalid_pairs": int((adj["relation"].isna() | (adj["valid_json"] == False)).sum()),  # noqa: E712
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ["n_dropped", "labels", "labels_high_conf_only",
                       "still_in_record_share", "absent_share", "consolidation_reasons"]},
                     indent=2))
    print("\nby prior explanation:")
    print(pd.crosstab(labels["prior_explanation"], labels["final_label"]).to_string())


if __name__ == "__main__":
    main()
