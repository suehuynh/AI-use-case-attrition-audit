"""Phase 1 step 2: tag explained drops, then retrieve cross-agency candidates.

Reads outputs/baseline_dropped_2024.csv (from run_baseline.py).
Writes:
  outputs/dropped_tagged.csv          dropped records + prior_explanation column
  outputs/retrieval_candidates.csv    top-k 2025 candidates per dropped record (all 826)
  outputs/retrieval_sensitivity.csv   cross-agency share by agency weight
  outputs/retrieval_summary.json

Usage: python scripts/run_retrieval.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402
from attrition.retrieval import CandidateRetriever  # noqa: E402

log = logging.getLogger(__name__)


def tag_prior_explanations(dropped: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Label drops with an obvious explanation. Retrieval still runs on ALL drops:
    a retired or USAID record could have a successor elsewhere in 2025."""
    d = dropped.copy()
    dissolved = d["agency"].isin(cfg.dissolved_agencies)
    retired = d["stage"].str.strip().str.lower().eq("retired")
    d["prior_explanation"] = "none"
    d.loc[retired, "prior_explanation"] = "retired_in_2024"
    d.loc[dissolved, "prior_explanation"] = "dissolved_agency"
    d.loc[dissolved & retired, "prior_explanation"] = "dissolved_agency+retired_in_2024"
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = Config.load(args.config)
    out = cfg.output_dir
    nxt = InventoryLoader(cfg).load("2025")
    dropped = pd.read_csv(out / "baseline_dropped_2024.csv", dtype=str).fillna("")
    dropped = tag_prior_explanations(dropped, cfg)
    dropped.to_csv(out / "dropped_tagged.csv", index=False)

    # Main run
    retriever = CandidateRetriever(cfg)
    cands = retriever.retrieve(dropped, nxt)
    cands.to_csv(out / "retrieval_candidates.csv", index=False)

    # Sensitivity: how much does the agency bonus crowd out cross-agency candidates?
    sens = []
    for w in cfg.retrieval["agency_weight_grid"]:
        c = CandidateRetriever(cfg, agency_weight=w).retrieve(dropped, nxt)
        top1 = c[c["rank"] == 1]
        sens.append({
            "agency_weight": w,
            "cross_agency_share_topk": round(CandidateRetriever.cross_agency_share(c), 3),
            "cross_agency_share_top1": round(CandidateRetriever.cross_agency_share(top1), 3),
        })
    pd.DataFrame(sens).to_csv(out / "retrieval_sensitivity.csv", index=False)

    top1 = cands[cands["rank"] == 1].merge(
        dropped[["uid", "prior_explanation"]], left_on="uid_prev", right_on="uid")
    summary = {
        "n_dropped": len(dropped),
        "prior_explanation_counts": dropped["prior_explanation"].value_counts().to_dict(),
        "n_candidate_pairs": len(cands),
        "cross_agency_share_topk": round(CandidateRetriever.cross_agency_share(cands), 3),
        "top1_score_quantiles": top1["score"].quantile([.1, .25, .5, .75, .9]).round(3).to_dict(),
        "sensitivity": sens,
    }
    (out / "retrieval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()