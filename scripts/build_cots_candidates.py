"""Flag dropped 2024 records that reappear as a same-agency product in the 2025 COTS checklist.

In 2025 OMB moved commercial off-the-shelf AI into a separate checklist file
(45 agencies x 20 standard tasks, with the product each agency uses). A dropped 2024 record whose
name matches a product its own agency lists there was reported, just in the other file.

Matching is by name only (token_set_ratio >= cots.min_name_score), so the count is a lower bound.

Usage: python scripts/build_cots_candidates.py
Reads  data/raw/<cots.file>, outputs/dropped_tagged.csv, data/raw/<2024 file> (agency names)
Writes outputs/cots_candidates.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = Config.load(ap.parse_args().config)
    c = cfg.raw["cots"]
    cots = pd.read_csv(cfg.raw_dir / c["file"], dtype=str).fillna("")

    # Full agency name -> abbreviation, from the 2024 file
    raw24 = InventoryLoader(cfg).load_raw("2024")
    full2abr = {k.strip(): v for k, v in raw24[["Agency", "Agency Abbreviation"]]
                .drop_duplicates().values}
    overrides = c.get("agency_name_overrides", {})

    def abbr(name: str) -> str | None:
        if name in overrides:
            return overrides[name]
        if name in full2abr:
            return full2abr[name]
        m = process.extractOne(name, list(full2abr), scorer=fuzz.token_sort_ratio)
        return full2abr[m[0]] if m and m[1] >= 92 else None

    cots["abr"] = cots["Agency"].map(abbr)
    unmapped = sorted(cots.loc[cots["abr"].isna(), "Agency"].unique())
    used = cots[(cots["Agency Use (Y/N)?"] == "Y")
                & (cots["Name of Commercial Product or Service Used"] != "")]
    products = (used.groupby("abr")["Name of Commercial Product or Service Used"]
                .apply(lambda s: sorted(set(s))).to_dict())

    dropped = pd.read_csv(cfg.output_dir / "dropped_tagged.csv", dtype=str).fillna("")
    hits = []
    for r in dropped.itertuples(index=False):
        scored = [(fuzz.token_set_ratio(r.name_key, p.lower()), p) for p in products.get(r.agency, [])]
        best = max(scored, default=(0, ""))
        if best[0] >= c["min_name_score"]:
            hits.append({"uid": r.uid, "agency": r.agency, "name_2024": r.name,
                         "cots_product_2025": best[1], "score": round(best[0], 1),
                         "prior_explanation": r.prior_explanation})
    out = pd.DataFrame(hits, columns=["uid", "agency", "name_2024", "cots_product_2025", "score",
                                      "prior_explanation"])
    out.to_csv(cfg.output_dir / "cots_candidates.csv", index=False)
    print(f"{len(out)} dropped records match a same-agency 2025 COTS product")
    print(f"{cots['abr'].notna().groupby(cots['Agency']).first().sum()} of "
          f"{cots['Agency'].nunique()} COTS agencies mapped; unmapped: {len(unmapped)}")
    # An agency that did not map to a 2024 name filed no 2024 records, so it cannot have drops.
    if unmapped:
        print("   (unmapped agencies filed no 2024 records, so they cannot affect this count)")


if __name__ == "__main__":
    main()
