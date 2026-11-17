"""Build the blind hand-labeling workbook for the pilot records.

Blind = model verdicts are NOT shown, so the human labels are not anchored on the LLM.
Each record is one block: a shaded header row with the dropped 2024 record, then its
10 retrieval candidates in a random order. Fill in the yellow cells on the header row only.

Usage: python scripts/make_labeling_sheet.py [--force]
Reads outputs/pilot_records.csv (written by run_adjudication.py --pilot)
Writes outputs/labeling/pilot_labeling_sheet.xlsx and pilot_labeling_key.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.labeling import PILOT_INSTRUCTIONS, LabelingSheetBuilder, save_guarded  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing labeling sheet")
    args = ap.parse_args()
    cfg = Config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    out = cfg.output_dir
    pilot = pd.read_csv(out / "pilot_records.csv", dtype=str).fillna("")
    cands = pd.read_csv(out / "retrieval_candidates.csv")
    nxt = InventoryLoader(cfg).load("2025")

    builder = LabelingSheetBuilder(nxt, seed=cfg.raw["adjudication"]["seed"])
    wb, key = builder.build(pilot, cands, PILOT_INSTRUCTIONS)
    label_dir = out / "labeling"
    save_guarded(wb, label_dir / "pilot_labeling_sheet.xlsx", args.force)
    key.to_csv(label_dir / "pilot_labeling_key.csv", index=False)
    print(f"{len(pilot)} records, {len(key)} candidates -> {label_dir}")


if __name__ == "__main__":
    main()