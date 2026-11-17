"""Score each model's pilot run against coder1's pilot sheet and apply the model choice rule.

Usage
  python scripts/score_pilot.py             # all 50 records labeled; writes model_choice.json
  python scripts/score_pilot.py --preview   # partial labels allowed; no model choice written

Reads  paths.pilot_sheet (uids come from the sheet), paths.pilot_pairs (consistency check),
       paths.adjudication for pilot__<model>, one per model in config.yaml
Writes paths.pilot_scores, paths.pilot_errors (one per model), paths.model_choice
model_choice.json is the fixed decision for the full run: it is not overwritten without --force.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.adjudicator import run_name  # noqa: E402
from attrition.config import Config  # noqa: E402
from attrition.labeling import LabeledSheet  # noqa: E402
from attrition.scoring import PilotScorer, choose_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--preview", action="store_true", help="score partial labels; no model choice")
    ap.add_argument("--force", action="store_true", help="overwrite an existing model_choice.json")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    sheet = LabeledSheet.read(cfg.path("pilot_sheet"))
    sheet.check_pairs(pd.read_csv(cfg.path("pilot_pairs"), dtype=str))
    n_all, n_done = len(sheet.records), len(sheet.labeled)
    if n_done < n_all and not args.preview:
        sys.exit(f"{n_all - n_done} of {n_all} pilot records are not labeled yet. "
                 "Finish the sheet, or use --preview.")
    if not n_done:
        sys.exit("No labeled records yet.")
    scorer = PilotScorer(sheet)

    rows, missing = [], []
    for model in cfg.raw["adjudication"]["models"]:
        path = cfg.path("adjudication", run=run_name("pilot", model))
        if not path.exists():
            missing.append(str(path))
            continue
        s, errors = scorer.score(pd.read_csv(path, dtype={"uid_prev": str, "uid_next": str}))
        short = model.split("/")[-1]
        rows.append({"model": short, **s})
        errors.to_csv(cfg.path("pilot_errors", model=short), index=False)
    if missing:
        sys.exit("missing pilot runs (copy them from Oscar):\n  " + "\n  ".join(missing))

    scores = pd.DataFrame(rows)
    print(scores.to_string(index=False))
    if args.preview:
        print(f"\nPREVIEW on {n_done} of {n_all} records: nothing written.")
        return
    scores.to_csv(cfg.path("pilot_scores"), index=False)

    choice_path = cfg.path("model_choice")
    if choice_path.exists() and not args.force:
        print(f"\n{choice_path} already exists; kept as is (use --force to redo the choice).")
        return
    choice = choose_model(scores)
    choice["scores"] = scores.to_dict("records")
    choice_path.write_text(json.dumps(choice, indent=2, default=str))
    print(f"\nModel choice: {choice['chosen']} ({choice['reason']}) -> {choice_path}")


if __name__ == "__main__":
    main()
