"""Replicate the 2024 -> 2025 baseline match and compare to the paper's counts.

Usage: python scripts/run_baseline.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.baseline import BaselineMatcher  # noqa: E402
from attrition.config import Config  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = Config.load(args.config)
    loader = InventoryLoader(cfg)
    prev, nxt = loader.load("2024"), loader.load("2025")
    result = BaselineMatcher(cfg).match(prev, nxt)

    summary = {"n_2024": len(prev), "n_2025": len(nxt), **result.summary()}
    summary["dropped_in_dissolved_agencies"] = int(
        result.dropped["agency"].isin(cfg.dissolved_agencies).sum()
    )
    print(json.dumps({"ours": summary, "paper": cfg.targets}, indent=2))

    out = cfg.output_dir
    result.matches.to_csv(out / "baseline_matches.csv", index=False)
    result.dropped.to_csv(out / "baseline_dropped_2024.csv", index=False)
    result.new.to_csv(out / "baseline_new_2025.csv", index=False)
    (out / "baseline_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()