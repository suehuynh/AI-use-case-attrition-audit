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

RELATIONS = "SAME_SYSTEM,PART_OF,SAME_PRODUCT_DIFFERENT_DEPLOYMENT,NO_MATCH"
HEADER_FILL = PatternFill("solid", fgColor="DCE6F1")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
WRAP = Alignment(wrap_text=True, vertical="top")

COLUMNS = ["record_uid", "role", "cand_id", "agency", "bureau", "name", "purpose",
           "match_cand_ids", "relation", "confidence", "notes"]
WIDTHS = [12, 9, 8, 9, 18, 34, 70, 14, 26, 12, 30]

INSTRUCTIONS = [
    "HOW TO LABEL (fill the yellow cells on each blue record row only)",
    "1. Read the 2024 record, then its 10 candidates (shown in random order).",
    "2. match_cand_ids: the cand_id of the matching candidate, e.g. 'C3'. "
    "Use commas for more than one, e.g. 'C3,C7'. Leave empty if none match.",
    "3. relation: SAME_SYSTEM (same system; rename or agency move is fine), "
    "PART_OF (2024 record absorbed into a broader 2025 entry), "
    "SAME_PRODUCT_DIFFERENT_DEPLOYMENT (same commercial tool, separate deployment), "
    "NO_MATCH (none of the 10).",
    "4. confidence: high / medium / low.",
    "Judge from the text only. Do not look at the model outputs until all 50 are done.",
]


def main() -> None:
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
    ws.add_data_validation(dv_rel)
    ws.add_data_validation(dv_conf)

    key_rows = []
    r = hdr_row + 1
    for rec in pilot.itertuples(index=False):
        rec = rec._asdict()
        vals = [rec["uid"], "2024", "", rec["agency"], rec["bureau"], rec["name"],
                rec["purpose"], "", "", "", ""]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=r, column=j, value=v)
            c.alignment = WRAP
            c.fill = INPUT_FILL if j >= 8 else HEADER_FILL
            if j <= 7:
                c.font = Font(bold=True)
        dv_rel.add(ws.cell(row=r, column=9))
        dv_conf.add(ws.cell(row=r, column=10))
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
    wb.save(label_dir / "pilot_labeling_sheet.xlsx")
    # Hidden key: maps C1..C10 back to 2025 uids and retrieval rank. Not for the labeler.
    pd.DataFrame(key_rows).to_csv(label_dir / "pilot_labeling_key.csv", index=False)
    print(f"{len(pilot)} records, {len(key_rows)} candidates -> {label_dir}")


if __name__ == "__main__":
    main()