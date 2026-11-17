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
from attrition.sampling import ValidationSampler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing sheets")
    ap.add_argument("--only", nargs="*", default=None, metavar="CODER",
                    help="rebuild only these coders' sheets; the sample must match the saved manifest")
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

    manifest = ValidationSampler(v["allocation"], v["seed"]).sample(labels, exclude=pilot)
    # Coders see records in a shuffled order so strata are not grouped together
    order = manifest.sample(frac=1, random_state=v["seed"])["uid"].tolist()
    records = dropped.set_index("uid").loc[order].reset_index()

    inventory = nxt[["uid", "agency", "bureau", "name", "purpose"]].copy()
    inventory["purpose"] = inventory["purpose"].str.slice(0, 300)
    builder = LabelingSheetBuilder(nxt, seed=v["seed"])
    extra = [("found_2025_uid", 16)]

    cols = ["uid", "agency", "stratum", "N_h", "n_h", "weight", "final_label"]
    coders = args.only if args.only else v["coders"]
    key = None
    for coder in coders:
        wb, key = builder.build(records, cands, VALIDATION_INSTRUCTIONS, extra_columns=extra,
                                inventory_tab=inventory)
        if args.only:
            # Rebuilding one sheet must not change the sample or the candidate order
            old_m = pd.read_csv(vdir / "sample_manifest.csv", dtype=str)
            old_k = pd.read_csv(vdir / "validation_key.csv", dtype=str)
            same_m = sorted(old_m["uid"]) == sorted(manifest["uid"])
            same_k = old_k.reset_index(drop=True).equals(key.astype(str).reset_index(drop=True))
            if not (same_m and same_k):
                sys.exit("STOP: the rebuilt sample or candidate order differs from the saved one; "
                         "nothing was written.")
        save_guarded(wb, vdir / f"validation_sheet__{coder}.xlsx", args.force)
    if not args.only:
        key.to_csv(vdir / "validation_key.csv", index=False)
        manifest[cols].to_csv(vdir / "sample_manifest.csv", index=False)
    else:
        print("sample and candidate order unchanged; manifest and key kept as they were")

    print(f"{len(manifest)} records sampled from {labels.shape[0] - len(pilot)} non-pilot records")
    print(manifest.groupby("stratum")[["N_h", "n_h"]].first().to_string())
    print(f"sheets for {', '.join(coders)} -> {vdir}")


if __name__ == "__main__":
    main()
