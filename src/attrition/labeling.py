"""Blind labeling workbooks (pilot and Phase 2 validation).

Each record is one block: a blue header row with the dropped 2024 record, then its retrieval
candidates in a seeded random order (so rank does not hint at the answer). Coders fill the
yellow cells on the header row. Model verdicts are never shown.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

RELATIONS = "SAME_SYSTEM,PART_OF,SPLIT_INTO,SAME_PRODUCT_DIFFERENT_DEPLOYMENT,NO_MATCH"
EXTERNAL = ("continued,expanded,replaced,transferred_federal,transferred_out,"
            "in_agency_inventory,paused,sunset,unknown")
HEADER_FILL = PatternFill("solid", fgColor="DCE6F1")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
WRAP = Alignment(wrap_text=True, vertical="top")
HEADER_ROW = 8  # instructions occupy rows 1-6; column names on row 8

BASE_COLUMNS = ["record_uid", "role", "cand_id", "agency", "bureau", "name", "purpose",
                "match_cand_ids", "relation", "confidence", "notes"]
BASE_WIDTHS = [12, 9, 8, 9, 18, 34, 70, 14, 26, 12, 30]

PILOT_INSTRUCTIONS = [
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

VALIDATION_INSTRUCTIONS = [
    "VALIDATION LABELING: code independently; do not discuss records with the other coder until "
    "both sheets are done. Rules: docs/codebook.md (Q1 only; use the inventory text, no web search).",
    "1. Read the 2024 record, then its 10 candidates (random order). Fill the yellow cells on the "
    "blue row: match_cand_ids, relation, confidence.",
    "2. match_cand_ids: e.g. 'C3'. More than one only for SPLIT_INTO or an exact duplicate. Other "
    "agencies' copies of the same product are SAME_PRODUCT_DIFFERENT_DEPLOYMENT, not matches.",
    "3. If NO candidate matches, spend up to 2 minutes searching the '2025_inventory' tab (Ctrl+F on "
    "key words). If you find the same system there, put its uid (e.g. 2025-01234) in found_2025_uid "
    "and set relation to the relation with that entry.",
    "4. relation (one): SAME_SYSTEM, PART_OF, SPLIT_INTO, SAME_PRODUCT_DIFFERENT_DEPLOYMENT, NO_MATCH.",
    "Model outputs are hidden on purpose. Unsure? Pick the best label, set confidence to low, add a note.",
]


class LabelingSheetBuilder:
    def __init__(self, nxt: pd.DataFrame, seed: int):
        self.nxt = nxt.set_index("uid")
        self.seed = seed

    def build(self, records: pd.DataFrame, cands: pd.DataFrame, instructions: list[str],
              extra_columns: list[tuple[str, int]] = (), external_dropdown: bool = False,
              inventory_tab: pd.DataFrame | None = None) -> tuple[Workbook, pd.DataFrame]:
        """records: dropped 2024 rows (uid, agency, bureau, name, purpose).
        cands: retrieval candidates (uid_prev, uid_next, rank). Returns (workbook, key)."""
        columns = BASE_COLUMNS + [c for c, _ in extra_columns]
        widths = BASE_WIDTHS + [w for _, w in extra_columns]

        wb = Workbook()
        ws = wb.active
        ws.title = "labels"
        for i, line in enumerate(instructions, start=1):
            ws.cell(row=i, column=1, value=line).font = Font(bold=(i == 1))
        for j, (c, w) in enumerate(zip(columns, widths), start=1):
            ws.cell(row=HEADER_ROW, column=j, value=c).font = Font(bold=True)
            ws.column_dimensions[ws.cell(row=HEADER_ROW, column=j).column_letter].width = w
        ws.freeze_panes = ws.cell(row=HEADER_ROW + 1, column=1)

        dv_rel = DataValidation(type="list", formula1=f'"{RELATIONS}"', allow_blank=True)
        dv_conf = DataValidation(type="list", formula1='"high,medium,low"', allow_blank=True)
        ws.add_data_validation(dv_rel)
        ws.add_data_validation(dv_conf)
        dv_ext = None
        if external_dropdown:
            dv_ext = DataValidation(type="list", formula1=f'"{EXTERNAL}"', allow_blank=True)
            ws.add_data_validation(dv_ext)
        col = {c: j for j, c in enumerate(columns, start=1)}

        key_rows, r = [], HEADER_ROW + 1
        for rec in records.itertuples(index=False):
            rec = rec._asdict()
            vals = [rec["uid"], "2024", "", rec["agency"], rec["bureau"], rec["name"],
                    rec["purpose"]] + [""] * (len(columns) - 7)
            for j, v in enumerate(vals, start=1):
                c = ws.cell(row=r, column=j, value=v)
                c.alignment = WRAP
                c.fill = INPUT_FILL if j >= col["match_cand_ids"] else HEADER_FILL
                if j <= 7:
                    c.font = Font(bold=True)
            dv_rel.add(ws.cell(row=r, column=col["relation"]))
            dv_conf.add(ws.cell(row=r, column=col["confidence"]))
            if dv_ext is not None and "external_status" in col:
                dv_ext.add(ws.cell(row=r, column=col["external_status"]))
            r += 1

            cc = cands[cands["uid_prev"] == rec["uid"]].sample(frac=1, random_state=self.seed)
            for k, cand in enumerate(cc.itertuples(), start=1):
                b = self.nxt.loc[cand.uid_next]
                cid = f"C{k}"
                key_rows.append({"record_uid": rec["uid"], "cand_id": cid,
                                 "uid_next": cand.uid_next, "rank": cand.rank})
                for j, v in enumerate(["", "2025", cid, b["agency"], b["bureau"], b["name"],
                                       b["purpose"]], start=1):
                    ws.cell(row=r, column=j, value=v).alignment = WRAP
                r += 1
            r += 1

        if inventory_tab is not None:
            inv = wb.create_sheet("2025_inventory")
            cols = list(inventory_tab.columns)
            for j, c in enumerate(cols, start=1):
                inv.cell(row=1, column=j, value=c).font = Font(bold=True)
            for i, row in enumerate(inventory_tab.itertuples(index=False), start=2):
                for j, v in enumerate(row, start=1):
                    inv.cell(row=i, column=j, value=v)
            for j, w in enumerate([12, 9, 30, 60, 80][: len(cols)], start=1):
                inv.column_dimensions[inv.cell(row=1, column=j).column_letter].width = w
            inv.freeze_panes = "A2"
            inv.auto_filter.ref = inv.dimensions
        return wb, pd.DataFrame(key_rows)


def save_guarded(wb: Workbook, path: Path, force: bool) -> None:
    """Refuse to overwrite a sheet that may already hold human labels."""
    if path.exists() and not force:
        raise SystemExit(f"{path} exists and may hold labels. Rerun with --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)