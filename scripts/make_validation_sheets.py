"""Draw the validation sample and build one blind sheet per coder.

Usage: python scripts/make_validation_sheets.py [--force]

Strata come from the final labels of the chosen model's full run (model_choice.json).
Reads  outputs/final/full__<chosen>/record_labels.csv, outputs/retrieval_candidates.csv,
       outputs/dropped_tagged.csv, paths.pilot_records (excluded from the sample)
Writes paths.validation_manifest  uid, agency, stratum, N_h, n_h, weight, final_label
                                  (model labels: never show this file to a coder)
       paths.validation_pairs     record_uid, cand_id, uid_next, rank
       paths.validation_sheet     one identical sheet per coder, each in its own folder
                                  (outputs/handoff/<coder>/); nothing else goes in those folders
Refuses to overwrite existing sheets without --force (they may hold labels).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.labeling import VALIDATION_TITLE, LabelingSheetBuilder, save_guarded  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402
from attrition.sampling import ValidationSampler  # noqa: E402
from attrition.scoring import chosen_full_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite existing sheets")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    v = cfg.raw["validation"]
    sheets = {c: cfg.path("validation_sheet", coder=c) for c in v["coders"]}
    existing = [str(p) for p in sheets.values() if p.exists()]
    if existing and not args.force:
        sys.exit("sheets exist and may hold labels (rerun with --force to overwrite):\n  "
                 + "\n  ".join(existing))

    out = cfg.output_dir
    run = chosen_full_run(cfg.path("model_choice"))
    labels = pd.read_csv(out / "final" / run / "record_labels.csv", dtype=str).fillna("")
    dropped = pd.read_csv(out / "dropped_tagged.csv", dtype=str).fillna("")
    cands = pd.read_csv(out / "retrieval_candidates.csv", dtype={"uid_prev": str, "uid_next": str})
    pilot = set(pd.read_csv(cfg.path("pilot_records"), dtype=str)["uid"])

    manifest = ValidationSampler(v["allocation"], v["seed"]).sample(labels, exclude=pilot)
    # Coders see records in a shuffled order so strata are not grouped together
    order = manifest.sample(frac=1, random_state=v["seed"])["uid"].tolist()
    records = dropped.set_index("uid").loc[order].reset_index()

    loader = InventoryLoader(cfg)
    builder = LabelingSheetBuilder(loader.load("2025"), loader.load("2024"), seed=v["seed"])
    wb, pairs = builder.build(records, cands, VALIDATION_TITLE)
    for path in sheets.values():
        save_guarded(wb, path, force=True)
    pairs.to_csv(cfg.path("validation_pairs"), index=False)
    cols = ["uid", "agency", "stratum", "N_h", "n_h", "weight", "final_label"]
    manifest[cols].to_csv(cfg.path("validation_manifest"), index=False)

    print(f"{len(manifest)} records sampled from {len(labels) - len(pilot)} non-pilot records "
          f"(strata from {run})")
    print(manifest.groupby("stratum")[["N_h", "n_h"]].first().to_string())
    for c, p in sheets.items():
        print(f"  {c}: {p}")


if __name__ == "__main__":
    main()
