"""Score each model's pilot run against the human labels.

Reads
  outputs/labeling/pilot_labeling_sheet.xlsx   (filled in by hand)
  outputs/labeling/pilot_labeling_key.csv      (maps C1..C10 -> 2025 uid)
  outputs/adjudication/pilot__*.csv            (one file per model)
Writes
  outputs/labeling/pilot_scores.csv            one row per model
  outputs/labeling/pilot_errors__<model>.csv   pairs where model and human disagree

Pair level: a pair is a "match" if its relation is SAME_SYSTEM, PART_OF or SPLIT_INTO.
  SAME_PRODUCT_DIFFERENT_DEPLOYMENT and DIFFERENT both count as "no match".
Record level: the model's pick is its best match candidate (confidence, then retrieval rank);
  the record is correct if the pick equals the human's match, or both say no match.

Usage: python scripts/score_pilot.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attrition.config import Config  # noqa: E402

MATCH = {"SAME_SYSTEM", "PART_OF", "SPLIT_INTO"}
CONF_ORDER = {"high": 0, "medium": 1, "low": 2}


def load_human(label_dir: Path) -> pd.DataFrame:
    """One row per (record, candidate) with the human relation."""
    sheet = pd.read_excel(label_dir / "pilot_labeling_sheet.xlsx", header=7, dtype=str).fillna("")
    rec = sheet[sheet["role"] == "2024"].copy()
    unlabeled = rec[rec["relation"].str.strip() == ""]
    if len(unlabeled):
        print(f"WARNING: {len(unlabeled)} of {len(rec)} records have no relation yet; "
              "they are excluded from scoring.")
    rec = rec[rec["relation"].str.strip() != ""]

    key = pd.read_csv(label_dir / "pilot_labeling_key.csv", dtype={"rank": int})
    rows = []
    for r in rec.itertuples():
        ids = {c.upper() for c in re.split(r"[,\s;]+", str(r.match_cand_ids)) if c}
        rel = r.relation.strip()
        for k in key[key["record_uid"] == r.record_uid].itertuples():
            human = rel if (k.cand_id in ids and rel != "NO_MATCH") else "NO_MATCH"
            rows.append({"uid_prev": r.record_uid, "uid_next": k.uid_next, "cand_id": k.cand_id,
                         "human_relation": human, "human_confidence": r.confidence})
    return pd.DataFrame(rows)


def model_pick(g: pd.DataFrame) -> str | None:
    m = g[g["relation"].isin(MATCH)].copy()
    if m.empty:
        return None
    m["c"] = m["confidence"].map(CONF_ORDER).fillna(3)
    return m.sort_values(["c", "rank"]).iloc[0]["uid_next"]


def score(human: pd.DataFrame, model: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = human.merge(model[["uid_prev", "uid_next", "rank", "relation", "confidence",
                            "quote_2024_found", "quote_2025_found", "rationale"]],
                     on=["uid_prev", "uid_next"], how="left")
    df["relation"] = df["relation"].fillna("INVALID")
    h = df["human_relation"].isin(MATCH)
    p = df["relation"].isin(MATCH)
    tp, fp, fn = int((h & p).sum()), int((~h & p).sum()), int((h & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")

    # Multi-way: SAME_SYSTEM / PART_OF / SPLIT_INTO / NONE
    three = lambda s: s.where(s.isin(MATCH), "NONE")  # noqa: E731
    kappa = cohen_kappa_score(three(df["human_relation"]), three(df["relation"]))

    rec_rows = []
    for uid, g in df.groupby("uid_prev"):
        hm = g.loc[g["human_relation"].isin(MATCH), "uid_next"].tolist()
        pick = model_pick(g)
        ok = (pick is None and not hm) or (pick in hm)
        rec_rows.append(ok)

    s = {
        "pairs": len(df),
        "records": df["uid_prev"].nunique(),
        "human_matches": int(h.sum()),
        "model_matches": int(p.sum()),
        "pair_precision": round(prec, 3),
        "pair_recall": round(rec, 3),
        "pair_f1": round(2 * prec * rec / (prec + rec), 3) if tp else 0.0,
        "pair_kappa_3way": round(kappa, 3),
        "record_accuracy": round(sum(rec_rows) / len(rec_rows), 3),
    }
    errors = df[h != p].sort_values(["uid_prev", "rank"])
    return s, errors


def external_summary(label_dir: Path) -> None:
    """Q2 counts for NO_MATCH records; a code without a source_url is reported as unverified."""
    sheet = pd.read_excel(label_dir / "pilot_labeling_sheet.xlsx", header=7, dtype=str).fillna("")
    if "external_status" not in sheet.columns:
        return
    rec = sheet[(sheet["role"] == "2024") & (sheet["relation"] == "NO_MATCH")]
    has = rec["external_status"].str.strip() != ""
    verified = has & (rec["source_url"].str.strip() != "")
    print(f"\nQ2 external status for {len(rec)} NO_MATCH records "
          f"({int(verified.sum())} with a source URL, {int((has & ~verified).sum())} unverified):")
    tab = rec[has].assign(verified=verified[has].map({True: "verified", False: "unverified"}))
    if len(tab):
        print(pd.crosstab(tab["external_status"], tab["verified"]).to_string())


def main() -> None:
    cfg = Config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    label_dir = cfg.output_dir / "labeling"
    human = load_human(label_dir)
    if human.empty:
        sys.exit("No labeled records yet.")

    rows = []
    for f in sorted((cfg.output_dir / "adjudication").glob("pilot__*.csv")):
        model = pd.read_csv(f)
        name = f.stem.split("__")[1]
        s, errors = score(human, model)
        rows.append({"model": name, **s})
        errors.to_csv(label_dir / f"pilot_errors__{name}.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(label_dir / "pilot_scores.csv", index=False)
    print(scores.to_string(index=False))
    external_summary(label_dir)


if __name__ == "__main__":
    main()