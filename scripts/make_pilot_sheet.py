"""Draw the 50-record pilot sample and build coder1's blind pilot sheet.

Usage: python scripts/make_pilot_sheet.py [--force]

Reads  outputs/dropped_tagged.csv, outputs/retrieval_candidates.csv
Writes paths.pilot_records   the 50 pilot records
       paths.pilot_pairs     record_uid, cand_id, uid_next, rank (what the models judge)
       paths.pilot_sheet     coder1's sheet (hidden uid_next on every candidate row)
Refuses to overwrite an existing sheet without --force (it may hold labels).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.labeling import PILOT_TITLE, LabelingSheetBuilder, save_guarded  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402
from attrition.sampling import PilotSampler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite an existing pilot sheet")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    p = cfg.raw["adjudication"]
    sheet_path = cfg.path("pilot_sheet")
    if sheet_path.exists() and not args.force:
        sys.exit(f"{sheet_path} exists and may hold labels. Rerun with --force to overwrite.")

    out = cfg.output_dir
    dropped = pd.read_csv(out / "dropped_tagged.csv", dtype=str).fillna("")
    cands = pd.read_csv(out / "retrieval_candidates.csv", dtype={"uid_prev": str, "uid_next": str})
    loader = InventoryLoader(cfg)
    pilot = PilotSampler.sample(dropped, p["pilot_n"], p["seed"])

    builder = LabelingSheetBuilder(loader.load("2025"), loader.load("2024"), seed=p["seed"])
    wb, pairs = builder.build(pilot, cands, PILOT_TITLE)
    save_guarded(wb, sheet_path, args.force)
    pilot.to_csv(cfg.path("pilot_records"), index=False)
    pairs.to_csv(cfg.path("pilot_pairs"), index=False)
    print(f"{len(pilot)} records, {len(pairs)} pairs\n  sheet: {sheet_path}\n"
          f"  pairs: {cfg.path('pilot_pairs')}")


if __name__ == "__main__":
    main()
