"""Build a blind hand-labeling workbook for the pilot records.

Blind = model verdicts are NOT shown, so the human labels are not anchored on the LLM.

Each record is one block: a shaded header row with the dropped 2024 record, then its
10 retrieval candidates in a random order (so rank order does not hint at the answer).
Fill in the yellow cells on the header row only.

Usage: python scripts/make_labeling_sheet.py
Reads outputs/pilot_records.csv (written by run_adjudication.py --pilot)
Writes outputs/labeling/pilot_labeling_sheet.xlsx
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.loader import InventoryLoader  # noqa: E402

RELATIONS = "SAME_SYSTEM,PART_OF,SPLIT_INTO,SAME_PRODUCT_DIFFERENT_DEPLOYMENT,NO_MATCH"
EXTERNAL = ("continued,expanded,replaced,transferred_federal,transferred_out,"
            "in_agency_inventory,paused,sunset,unknown")
HEADER_FILL = PatternFill("solid", fgColor="DCE6F1")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
WRAP = Alignment(wrap_text=True, vertical="top")

COLUMNS = ["record_uid", "role", "cand_id", "agency", "bureau", "name", "purpose",
           "match_cand_ids", "relation", "confidence", "notes", "external_status", "source_url"]
WIDTHS = [12, 9, 8, 9, 18, 34, 70, 14, 26, 12, 30, 20, 40]

INSTRUCTIONS = [
    "HOW TO LABEL (fill the yellow cells on each blue record row only). Full rules: docs/codebook.md",
    "1. Read the 2024 record, then its 10 candidates (shown in random order). Use the text only.",
    "2. match_cand_ids: the matching candidate, e.g. 'C3'. More than one only for SPLIT_INTO or an "
    "exact duplicate. Other agencies' copies of the same product are NOT matches.",
    "3. relation (one): SAME_SYSTEM, PART_OF (absorbed into a broader 2025 entry), SPLIT_INTO "
    "(broad 2024 entry divided into specific 2025 entries), SAME_PRODUCT_DIFFERENT_DEPLOYMENT, "
    "NO_MATCH. confidence: high / medium / low.",
    "4. Only for NO_MATCH, optional: external_status + source_url (primary source required).",
    "Do not look at the model outputs until every record is labeled.",
]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing labeling sheet")
    args = ap.parse_args()
    cfg = Config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    out = cfg.output_dir
    seed = cfg.raw["adjudication"]["seed"]

    pilot = pd.read_csv(out / "pilot_records.csv", dtype=str).fillna("")
    cands = pd.read_csv(out / "retrieval_candidates.csv")
    nxt = InventoryLoader(cfg).load("2025").set_index("uid")

    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    for i, line in enumerate(INSTRUCTIONS, start=1):
        ws.cell(row=i, column=1, value=line).font = Font(bold=(i == 1))
    hdr_row = len(INSTRUCTIONS) + 2
    for j, (c, w) in enumerate(zip(COLUMNS, WIDTHS), start=1):
        ws.cell(row=hdr_row, column=j, value=c).font = Font(bold=True)
        ws.column_dimensions[chr(64 + j)].width = w
    ws.freeze_panes = ws.cell(row=hdr_row + 1, column=1)

    dv_rel = DataValidation(type="list", formula1=f'"{RELATIONS}"', allow_blank=True)
    dv_conf = DataValidation(type="list", formula1='"high,medium,low"', allow_blank=True)
    dv_ext = DataValidation(type="list", formula1=f'"{EXTERNAL}"', allow_blank=True)
    ws.add_data_validation(dv_rel)
    ws.add_data_validation(dv_conf)
    ws.add_data_validation(dv_ext)

    key_rows = []
    r = hdr_row + 1
    for rec in pilot.itertuples(index=False):
        rec = rec._asdict()
        vals = [rec["uid"], "2024", "", rec["agency"], rec["bureau"], rec["name"],
                rec["purpose"], "", "", "", "", "", ""]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=r, column=j, value=v)
            c.alignment = WRAP
            c.fill = INPUT_FILL if j >= 8 else HEADER_FILL
            if j <= 7:
                c.font = Font(bold=True)
        dv_rel.add(ws.cell(row=r, column=9))
        dv_conf.add(ws.cell(row=r, column=10))
        dv_ext.add(ws.cell(row=r, column=12))
        r += 1

        cc = cands[cands["uid_prev"] == rec["uid"]].sample(frac=1, random_state=seed)
        for k, cand in enumerate(cc.itertuples(), start=1):
            b = nxt.loc[cand.uid_next]
            cid = f"C{k}"
            key_rows.append({"record_uid": rec["uid"], "cand_id": cid,
                             "uid_next": cand.uid_next, "rank": cand.rank})
            vals = ["", "2025", cid, b["agency"], b["bureau"], b["name"], b["purpose"]]
            for j, v in enumerate(vals, start=1):
                ws.cell(row=r, column=j, value=v).alignment = WRAP
            r += 1
        r += 1  # blank line between records

    label_dir = out / "labeling"
    label_dir.mkdir(exist_ok=True)
    target = label_dir / "pilot_labeling_sheet.xlsx"
    if target.exists() and not args.force:
        sys.exit(f"{target} exists and may hold your labels. Rerun with --force to overwrite.")
    wb.save(target)
    # Hidden key: maps C1..C10 back to 2025 uids and retrieval rank. Not for the labeler.
    pd.DataFrame(key_rows).to_csv(label_dir / "pilot_labeling_key.csv", index=False)
    print(f"{len(pilot)} records, {len(key_rows)} candidates -> {label_dir}")


if __name__ == "__main__":
    main()