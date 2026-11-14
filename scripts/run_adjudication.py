"""Run pairwise LLM adjudication on retrieval candidates.

Examples
  # Laptop: check prompts and the pipeline with no model
  python scripts/run_adjudication.py --backend mock --pilot
  python scripts/run_adjudication.py --dump-prompts 3

  # HPC: pilot (50 dropped records x 10 candidates = 500 calls)
  python scripts/run_adjudication.py --model meta-llama/Llama-3.1-8B-Instruct --pilot

  # HPC: full run (826 x 10 = 8,260 calls)
  python scripts/run_adjudication.py --model meta-llama/Llama-3.1-8B-Instruct

Writes outputs/adjudication/<run_name>.csv and a small JSON summary next to it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.adjudicator import LLMAdjudicator, PilotSampler  # noqa: E402
from attrition.config import Config  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="HF model id; default = first model in config")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "transformers", "mock"])
    ap.add_argument("--pilot", action="store_true", help="run on the pilot sample only")
    ap.add_argument("--dump-prompts", type=int, default=0, help="print N prompts and exit")
    ap.add_argument("--prompt-file", default=None,
                    help="override config adjudication.prompt_file, e.g. prompts/adjudicator_v1.txt")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.prompt_file:
        cfg.raw["adjudication"]["prompt_file"] = args.prompt_file
    p = cfg.raw["adjudication"]
    model = args.model or p["models"][0]
    out = cfg.output_dir

    nxt = InventoryLoader(cfg).load("2025")
    dropped = pd.read_csv(out / "dropped_tagged.csv", dtype=str).fillna("")
    cands = pd.read_csv(out / "retrieval_candidates.csv")

    if args.pilot:
        pilot = PilotSampler.sample(dropped, p["pilot_n"], p["seed"])
        pilot.to_csv(out / "pilot_records.csv", index=False)
        cands = cands[cands["uid_prev"].isin(pilot["uid"])]
    print(f"model={model} backend={args.backend} pairs={len(cands)}")

    adj = LLMAdjudicator(cfg, model, args.backend)

    if args.dump_prompts:
        P, N = dropped.set_index("uid"), nxt.set_index("uid")
        for r in cands.head(args.dump_prompts).itertuples():
            for m in adj.prompts.messages(P.loc[r.uid_prev], N.loc[r.uid_next]):
                if m["role"] == "user":
                    print("-" * 80, "\n", m["content"])
        print("-" * 80, f"\nsystem prompt: {len(adj.prompts.system)} chars")
        return

    t0 = time.time()
    res = adj.adjudicate(cands, dropped, nxt)
    elapsed = round(time.time() - t0, 1)

    run_name = f"{'pilot' if args.pilot else 'full'}__{model.split('/')[-1]}__{adj.prompts.version}"
    run_dir = out / "adjudication"
    run_dir.mkdir(exist_ok=True)
    res.to_csv(run_dir / f"{run_name}.csv", index=False)

    summary = {
        "run": run_name, "model": model, "backend": adj.backend_version,
        "pairs": len(res), "seconds": elapsed,
        "valid_json_rate": round(res["valid_json"].mean(), 3),
        "quote_2024_found_rate": round(res["quote_2024_found"].mean(), 3),
        "quote_2025_found_rate": round(res["quote_2025_found"].mean(), 3),
        "relation_counts": res["relation"].fillna("INVALID").value_counts().to_dict(),
    }
    (run_dir / f"{run_name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()