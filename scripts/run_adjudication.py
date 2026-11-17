"""Pairwise LLM adjudication (SAME_SYSTEM / DIFFERENT) on a fixed set of pairs.

Examples
  # Laptop: check prompts and the pipeline with no model
  python scripts/run_adjudication.py --stage pilot --backend mock
  python scripts/run_adjudication.py --stage pilot --dump-prompts 3

  # Oscar (see oscar/adjudicate.sbatch)
  python scripts/run_adjudication.py --stage pilot --model Qwen/Qwen2.5-7B-Instruct
  python scripts/run_adjudication.py --stage full  --model Qwen/Qwen2.5-7B-Instruct

Pairs judged
  pilot  exactly the pairs in paths.pilot_pairs (written with the pilot sheet), or --pairs FILE
  full   all retrieval candidates (826 x 10), or --pairs FILE
Writes paths.adjudication with run = <stage>__<model>, plus a JSON summary next to it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.adjudicator import LLMAdjudicator, run_name  # noqa: E402
from attrition.config import Config  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402


def load_pairs(cfg: Config, stage: str, pairs_file: str | None) -> pd.DataFrame:
    if pairs_file or stage == "pilot":
        path = Path(pairs_file) if pairs_file else cfg.path("pilot_pairs")
        if not path.exists():
            sys.exit(f"{path} not found: build the pilot sheet first (scripts/make_pilot_sheet.py).")
        pairs = pd.read_csv(path, dtype={"record_uid": str, "uid_next": str})
        return pairs.rename(columns={"record_uid": "uid_prev"})
    return pd.read_csv(cfg.output_dir / "retrieval_candidates.csv",
                       dtype={"uid_prev": str, "uid_next": str})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", required=True, choices=["pilot", "full"])
    ap.add_argument("--model", default=None, help="HF model id; default = first model in config")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "transformers", "mock"])
    ap.add_argument("--pairs", default=None, help="CSV with record_uid, uid_next, rank")
    ap.add_argument("--dump-prompts", type=int, default=0, help="print N prompts and exit")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    model = args.model or cfg.raw["adjudication"]["models"][0]
    nxt = InventoryLoader(cfg).load("2025")
    dropped = pd.read_csv(cfg.output_dir / "dropped_tagged.csv", dtype=str).fillna("")
    pairs = load_pairs(cfg, args.stage, args.pairs)
    print(f"stage={args.stage} model={model} backend={args.backend} pairs={len(pairs)}")

    adj = LLMAdjudicator(cfg, model, args.backend)
    if args.dump_prompts:
        P, N = dropped.set_index("uid"), nxt.set_index("uid")
        for r in pairs.head(args.dump_prompts).itertuples():
            for m in adj.prompts.messages(P.loc[r.uid_prev], N.loc[r.uid_next]):
                if m["role"] == "user":
                    print("-" * 80, "\n", m["content"])
        print("-" * 80, f"\nsystem prompt: {len(adj.prompts.system)} chars")
        return

    t0 = time.time()
    res = adj.adjudicate(pairs, dropped, nxt)
    name = run_name(args.stage, model)
    path = cfg.path("adjudication", run=name)
    res.to_csv(path, index=False)
    summary = {
        "run": name, "model": model, "backend": adj.backend_version, "pairs": len(res),
        "seconds": round(time.time() - t0, 1),
        "valid_json_rate": round(res["valid_json"].mean(), 3),
        "quote_2024_found_rate": round(res["quote_2024_found"].mean(), 3),
        "quote_2025_found_rate": round(res["quote_2025_found"].mean(), 3),
        "relation_counts": res["relation"].fillna("INVALID").value_counts().to_dict(),
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
