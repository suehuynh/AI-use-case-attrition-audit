"""Blind labeling workbooks (pilot and validation) that carry their own key.

Layout of the 'labels' tab (column names on HEADER_ROW):
  - one blue record row per dropped 2024 record (role = 2024); the coder fills the yellow cells
  - its retrieval candidates below it (role = 2025), in a seeded random order with ids C1..C10,
    so rank does not hint at the answer
  - EVERY row carries record_uid, and every candidate row carries the 2025 uid in the hidden
    column uid_next. The sheet is therefore its own key: reading it never depends on row order.

Each sheet is written together with a pairs file (record_uid, cand_id, uid_next, rank). The pairs
file is what the model judges; the reader checks that the sheet still matches it.
Model verdicts are never shown in a sheet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

RELATIONS = ["SAME_SYSTEM", "NO_MATCH"]
CONFIDENCES = ["high", "medium", "low"]
HEADER_ROW = 9  # instructions occupy rows 1-7; column names on row 9

COLUMNS = ["record_uid", "role", "cand_id", "agency", "bureau", "name", "purpose",
           "match_cand_ids", "relation", "confidence", "found_2025_uid", "notes", "uid_next"]
WIDTHS = [12, 6, 7, 9, 18, 34, 70, 14, 15, 11, 15, 30, 12]
INPUT_COLUMNS = ["match_cand_ids", "relation", "confidence", "found_2025_uid", "notes"]
PAIR_COLUMNS = ["record_uid", "cand_id", "uid_next", "rank"]

HEADER_FILL = PatternFill("solid", fgColor="DCE6F1")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
WRAP = Alignment(wrap_text=True, vertical="top")
GREY = Font(color="A6A6A6")


def instructions(title: str) -> list[str]:
    return [
        f"{title}. Full rules and decision tree: docs/codebook.md. Model outputs are hidden on purpose.",
        "1. Read the blue 2024 record, then its candidates (random order). Judge from the inventory "
        "text only; no web search.",
        "2. relation: SAME_SYSTEM if the 2024 system is still listed in 2025 (same or new name, inside a "
        "broader entry, or split into narrower entries); otherwise NO_MATCH.",
        "3. match_cand_ids (SAME_SYSTEM only): every matching candidate, e.g. 'C3' or 'C2, C7' for a "
        "split. Other agencies' copies of the same product are NOT matches.",
        "4. No candidate matches? Spend up to 2 minutes in the '2025_inventory' tab (Ctrl+F). If you "
        "find the system, put its uid (e.g. 2025-01234) in found_2025_uid and choose SAME_SYSTEM.",
        "5. confidence: high / medium / low. Unsure? Pick the better label, set low, add a note. "
        "'Previously named X' notes: look X up in the '2024_inventory' tab.",
        "Fill only the yellow cells on blue rows. Do not edit, sort or delete other cells.",
    ]


PILOT_TITLE = "PILOT LABELING (coder1)"
VALIDATION_TITLE = ("VALIDATION LABELING: code independently and do not discuss records with the "
                    "other coder until both sheets are done")


def _ids(text: str) -> set[str]:
    return {c.upper() for c in re.split(r"[,\s;]+", str(text)) if c}


# ---------------------------------------------------------------------------
class LabelingSheetBuilder:
    def __init__(self, nxt: pd.DataFrame, prev: pd.DataFrame, seed: int):
        self.nxt = nxt.set_index("uid")
        self.prev = prev
        self.seed = seed

    @staticmethod
    def _lookup_tab(wb: Workbook, title: str, df: pd.DataFrame) -> None:
        tab = df[["uid", "agency", "bureau", "name", "purpose"]].copy()
        tab["purpose"] = tab["purpose"].fillna("").str.slice(0, 300)
        ws = wb.create_sheet(title)
        for j, c in enumerate(tab.columns, start=1):
            ws.cell(row=1, column=j, value=c).font = Font(bold=True)
        for i, row in enumerate(tab.itertuples(index=False), start=2):
            for j, v in enumerate(row, start=1):
                ws.cell(row=i, column=j, value=v)
        for j, w in enumerate([12, 9, 30, 60, 80], start=1):
            ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = w
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

    def build(self, records: pd.DataFrame, cands: pd.DataFrame,
              title: str) -> tuple[Workbook, pd.DataFrame]:
        """records: dropped 2024 rows (uid, agency, bureau, name, purpose), in display order.
        cands: retrieval candidates (uid_prev, uid_next, rank). Returns (workbook, pairs)."""
        wb = Workbook()
        ws = wb.active
        ws.title = "labels"
        for i, line in enumerate(instructions(title), start=1):
            ws.cell(row=i, column=1, value=line).font = Font(bold=(i == 1))
        col = {c: j for j, c in enumerate(COLUMNS, start=1)}
        for c, j in col.items():
            ws.cell(row=HEADER_ROW, column=j, value=c).font = Font(bold=True)
            ws.column_dimensions[ws.cell(row=HEADER_ROW, column=j).column_letter].width = WIDTHS[j - 1]
        ws.column_dimensions[ws.cell(row=HEADER_ROW, column=col["uid_next"]).column_letter].hidden = True
        ws.freeze_panes = ws.cell(row=HEADER_ROW + 1, column=1)

        dv_rel = DataValidation(type="list", formula1=f'"{",".join(RELATIONS)}"', allow_blank=True)
        dv_conf = DataValidation(type="list", formula1=f'"{",".join(CONFIDENCES)}"', allow_blank=True)
        ws.add_data_validation(dv_rel)
        ws.add_data_validation(dv_conf)

        pairs, r = [], HEADER_ROW + 1
        for rec in records.itertuples(index=False):
            rec = rec._asdict()
            for c in COLUMNS:
                cell = ws.cell(row=r, column=col[c])
                cell.alignment = WRAP
                cell.fill = INPUT_FILL if c in INPUT_COLUMNS else HEADER_FILL
            for c, v in (("record_uid", rec["uid"]), ("role", "2024"), ("agency", rec["agency"]),
                         ("bureau", rec["bureau"]), ("name", rec["name"]), ("purpose", rec["purpose"])):
                ws.cell(row=r, column=col[c], value=v).font = Font(bold=True)
            dv_rel.add(ws.cell(row=r, column=col["relation"]))
            dv_conf.add(ws.cell(row=r, column=col["confidence"]))
            r += 1

            cc = cands[cands["uid_prev"] == rec["uid"]].sample(frac=1, random_state=self.seed)
            for k, cand in enumerate(cc.itertuples(), start=1):
                b = self.nxt.loc[cand.uid_next]
                cid = f"C{k}"
                pairs.append({"record_uid": rec["uid"], "cand_id": cid,
                              "uid_next": cand.uid_next, "rank": int(cand.rank)})
                vals = {"record_uid": rec["uid"], "role": "2025", "cand_id": cid,
                        "agency": b["agency"], "bureau": b["bureau"], "name": b["name"],
                        "purpose": b["purpose"], "uid_next": cand.uid_next}
                for c, v in vals.items():
                    cell = ws.cell(row=r, column=col[c], value=v)
                    cell.alignment = WRAP
                    if c == "record_uid":
                        cell.font = GREY
                r += 1
            r += 1

        self._lookup_tab(wb, "2025_inventory", self.nxt.reset_index())
        self._lookup_tab(wb, "2024_inventory", self.prev)
        return wb, pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def save_guarded(wb: Workbook, path: Path, force: bool) -> None:
    """Refuse to overwrite a sheet that may already hold human labels."""
    if path.exists() and not force:
        raise SystemExit(f"{path} exists and may hold labels. Rerun with --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# ---------------------------------------------------------------------------
@dataclass
class LabeledSheet:
    """A filled sheet: one row per record, one row per (record, candidate)."""
    records: pd.DataFrame      # index record_uid: agency, relation, match_cand_ids, confidence, found_2025_uid, notes
    candidates: pd.DataFrame   # record_uid, cand_id, uid_next

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "LabeledSheet":
        df = df.fillna("").astype(str).apply(lambda s: s.str.strip())
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"sheet is missing columns {missing}")
        rec = df[df["role"] == "2024"].set_index("record_uid")
        if rec.index.duplicated().any():
            raise ValueError(f"duplicate record rows: {sorted(rec.index[rec.index.duplicated()])}")
        cand = df.loc[df["role"] == "2025", ["record_uid", "cand_id", "uid_next"]]
        cand = cand.sort_values(["record_uid", "cand_id"]).reset_index(drop=True)
        sheet = cls(rec[["agency", "relation", "match_cand_ids", "confidence",
                         "found_2025_uid", "notes"]].sort_index(), cand)
        sheet.validate()
        return sheet

    @classmethod
    def read(cls, path: Path) -> "LabeledSheet":
        return cls.from_frame(pd.read_excel(path, sheet_name="labels", header=HEADER_ROW - 1, dtype=str))

    # ---- checks -------------------------------------------------------
    def validate(self) -> None:
        problems = []
        known = self.candidates.groupby("record_uid")["cand_id"].apply(set).to_dict()
        for uid, r in self.records.iterrows():
            if r["relation"] and r["relation"] not in RELATIONS:
                problems.append(f"{uid}: relation '{r['relation']}' is not one of {RELATIONS}")
            if r["relation"] == "SAME_SYSTEM":
                bad = _ids(r["match_cand_ids"]) - known.get(uid, set())
                if bad:
                    problems.append(f"{uid}: unknown candidate ids {sorted(bad)}")
                if not _ids(r["match_cand_ids"]) and not r["found_2025_uid"]:
                    problems.append(f"{uid}: SAME_SYSTEM but no match_cand_ids or found_2025_uid")
        if problems:
            raise ValueError("labeling sheet has problems:\n  " + "\n  ".join(problems))

    def check_pairs(self, pairs: pd.DataFrame) -> None:
        """The sheet must hold exactly the pairs in its pairs file."""
        key = lambda d: set(map(tuple, d[["record_uid", "cand_id", "uid_next"]].astype(str).values))  # noqa: E731
        a, b = key(self.candidates), key(pairs)
        if a != b:
            raise ValueError(f"sheet and pairs file differ: {len(a - b)} pairs only in the sheet, "
                             f"{len(b - a)} only in the pairs file")

    # ---- answers ------------------------------------------------------
    @property
    def labeled(self) -> list[str]:
        return self.records.index[self.records["relation"] != ""].tolist()

    def matched_uids(self, record_uid: str) -> set[str]:
        """2025 uids the coder says are the same system (empty for NO_MATCH)."""
        r = self.records.loc[record_uid]
        return self.uids_for(record_uid, r["relation"], r["match_cand_ids"], r["found_2025_uid"])

    def uids_for(self, record_uid: str, relation: str, match_cand_ids: str, found_uid: str) -> set[str]:
        """Resolve an answer (e.g. a resolved disagreement) to 2025 uids using this sheet's key."""
        if str(relation).strip() != "SAME_SYSTEM":
            return set()
        c = self.candidates[self.candidates["record_uid"] == record_uid]
        unknown = _ids(match_cand_ids) - set(c["cand_id"])
        if unknown:
            raise ValueError(f"{record_uid}: unknown candidate ids {sorted(unknown)}")
        uids = set(c.loc[c["cand_id"].isin(_ids(match_cand_ids)), "uid_next"])
        if str(found_uid).strip():
            uids.add(str(found_uid).strip())
        return uids

    def matched_cand_ids(self, record_uid: str) -> str:
        r = self.records.loc[record_uid]
        return ",".join(sorted(_ids(r["match_cand_ids"]))) if r["relation"] == "SAME_SYSTEM" else ""
