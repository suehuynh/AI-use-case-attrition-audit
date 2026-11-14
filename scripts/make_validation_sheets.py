"""Phase 2: draw the held-out validation sample and build one blind sheet per coder.

Usage: python scripts/make_validation_sheets.py [--force]

Reads  outputs/final/<validation.run>/record_labels.csv, outputs/retrieval_candidates.csv,
       outputs/dropped_tagged.csv, outputs/pilot_records.csv (excluded from the sample)
Writes outputs/validation/sample_manifest.csv     uid, stratum, N_h, n_h, weight (no model labels
                                                  for coders to see; keep this file to yourself)
       outputs/validation/validation_key.csv       C1..C10 -> 2025 uid
       outputs/validation/validation_sheet__<coder>.xlsx   identical content, one per coder
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.labeling import (VALIDATION_INSTRUCTIONS, LabelingSheetBuilder,  # noqa: E402
                                save_guarded)
from attrition.loader import InventoryLoader  # noqa: E402
from attrition.validation import ValidationSampler, assign_strata  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing sheets")
    args = ap.parse_args()

    cfg = Config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    v = cfg.raw["validation"]
    out = cfg.output_dir
    vdir = out / "validation"

    labels = pd.read_csv(out / "final" / v["run"] / "record_labels.csv", dtype=str).fillna("")
    dropped = pd.read_csv(out / "dropped_tagged.csv", dtype=str).fillna("")
    cands = pd.read_csv(out / "retrieval_candidates.csv")
    pilot = set(pd.read_csv(out / "pilot_records.csv", dtype=str)["uid"])
    nxt = InventoryLoader(cfg).load("2025")

    compare = None
    if v.get("compare_run"):
        cpath = out / "final" / v["compare_run"] / "record_labels.csv"
        if not cpath.exists():
            sys.exit(f"{cpath} not found: run run_classify.py for {v['compare_run']} first, "
                     "or remove validation.compare_run from config.yaml.")
        compare = pd.read_csv(cpath, dtype=str).fillna("")
    labels = assign_strata(labels, compare)
    manifest = ValidationSampler(v["allocation"], v["seed"]).sample(labels, exclude=pilot)
    # Coders see records in a shuffled order so strata are not grouped together
    order = manifest.sample(frac=1, random_state=v["seed"])["uid"].tolist()
    records = dropped.set_index("uid").loc[order].reset_index()

    inventory = nxt[["uid", "agency", "bureau", "name", "purpose"]].copy()
    inventory["purpose"] = inventory["purpose"].str.slice(0, 300)
    builder = LabelingSheetBuilder(nxt, seed=v["seed"])
    extra = [("found_2025_uid", 16)]

    for coder in v["coders"]:
        wb, key = builder.build(records, cands, VALIDATION_INSTRUCTIONS, extra_columns=extra,
                                inventory_tab=inventory)
        save_guarded(wb, vdir / f"validation_sheet__{coder}.xlsx", args.force)
    key.to_csv(vdir / "validation_key.csv", index=False)
    manifest[["uid", "agency", "stratum", "N_h", "n_h", "weight", "final_label"]].to_csv(
        vdir / "sample_manifest.csv", index=False)

    print(f"{len(manifest)} records sampled from {labels.shape[0] - len(pilot)} non-pilot records")
    print(manifest.groupby("stratum")[["N_h", "n_h"]].first().to_string())
    print(f"sheets for {', '.join(v['coders'])} -> {vdir}")


if __name__ == "__main__":
    main()
