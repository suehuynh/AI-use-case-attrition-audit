"""Apply the codebook-v2 review to coder 1's pilot labels, keeping every change auditable.

Input : outputs/labeling/human/pilot_labeling_sheet_coder1.xlsx   (original, never modified)
Output: outputs/labeling/pilot_labeling_sheet.xlsx              (reviewed; read by score_pilot.py)

Adds three columns: external_status, source_url, review_note.
Each entry in CHANGES / FLAGS / EXTERNAL states what changed and why (see docs/codebook.md).
To reverse a change, delete its entry here and rerun.
"""
from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/labeling/human/pilot_labeling_sheet_coder1.xlsx"
DST = ROOT / "outputs/labeling/pilot_labeling_sheet.xlsx"
HEADER_ROW = 8

RELATIONS = "SAME_SYSTEM,PART_OF,SPLIT_INTO,SAME_PRODUCT_DIFFERENT_DEPLOYMENT,NO_MATCH"
EXTERNAL = ("continued,expanded,replaced,transferred_federal,transferred_out,"
            "in_agency_inventory,paused,sunset,unknown")

HHS_PAGE_NOTE = ("HealthIT.gov pages with HHS-CMS/NIH IDs use the 2024 reporting format; HHS's "
                 "2025 inventory has no unique IDs (FedScoop). A recent page date is not "
                 "evidence of 2025 reporting. Set to unknown unless a 2025 source is found.")

# record_uid -> (match_cand_ids, relation, confidence or None to keep, reason)
CHANGES = {
    "2024-01993": ("C5", "SAME_SYSTEM", None,
                   "C5 is DOJ's own Westlaw entry (rename). C1/C3/C7/C8/C10 are TVA, OSHRC, SEC, "
                   "FHFA using the same product: SAME_PRODUCT, so removed from matches."),
    "2024-00577": ("C1, C8, C9", "SPLIT_INTO", None,
                   "Broad 2024 'AI tools in EPA's ServiceNow' entry split into specific EPA "
                   "entries (C1, C8, C9). C2/C4/C6/C7 are DOE, NEA, HHS ServiceNow: same product."),
    "2024-01488": ("", "NO_MATCH", None,
                   "Exiger candidates (C5, C9) are DOJ's deployment: same product, different "
                   "agency. USTDA's continued contract is external evidence (Q2), not linkage."),
    "2024-00713": ("", "NO_MATCH", None,
                   "No CMS candidate among the 10 and HHS-CMS-00020 is not in the 2025 OMB file."),
    "2024-00848": ("", "NO_MATCH", None,
                   "No candidate was listed; the NLM-Gene-2 expansion is external evidence (Q2)."),
    "2024-01129": ("", "NO_MATCH", "medium",
                   "All four candidates are other agencies' generic transcription tools, not the "
                   "Georgian program. Codebook: cross-agency SAME needs the same specific program."),
    "2024-01769": ("C1", "SAME_SYSTEM", "medium",
                   "C5 is HHS/CMS 'Docketscope' (different agency: same product), so removed. "
                   "Note said same agency, but only C1 is DOJ/ATF. CONFIRM C1 is DocketScope, "
                   "else set NO_MATCH."),
    "2024-00335": ("C9", "SAME_SYSTEM", None,
                   "Removed C7 (HHS SAM.gov RFI analysis: different agency and purpose)."),
    "2024-01315": ("C3", "PART_OF", None, "Removed C2 (FCC: different agency)."),
    "2024-01198": ("C9", "SAME_SYSTEM", None,
                   "Removed C2 (DOI) and C8 (DOJ): same task, other agencies."),
    "2024-01274": ("C9", "SAME_SYSTEM", None,
                   "Removed C3 (DOJ). C9 is also the match for 2024-01198: two 2024 records to one "
                   "2025 entry, which the aggregation rules label 'consolidated'."),
    "2024-00599": ("C9", "PART_OF", None,
                   "Removed C3 (FRB), C4 (SEC), C8 (TREAS): comment tools at other agencies."),
    "2024-01590": ("C2, C4, C9", "SPLIT_INTO", None,
                   "One broad 2024 VA entry, three specific 2025 VA entries: split. If only C9 "
                   "(VA Health Connect) is the same system, set C9 + SAME_SYSTEM instead."),
}

# Kept as labeled, but worth a second look
FLAGS = {
    "2024-01015": "CONFIRM: two IRS chatbots listed. If the voicebot became both, use SPLIT_INTO; "
                  "if one is the rename, keep only that one.",
    "2024-01649": "CONFIRM: C9 is a near-exact name match; C1 (Lyssn for Mental Health) looks like "
                  "a different tool. Consider C9 only.",
    "2024-00675": "CONFIRM: C8/C9 look like the enterprise ChatCDC; C1 (DGMH) and C2 (ATSDR) are "
                  "specialized chatbots. Consider C8, C9 only, or SPLIT_INTO.",
    "2024-01630": "NEEDS YOUR LABEL (left blank in the original).",
}

# record_uid -> (suggested external_status, reason). All need a source_url before use.
EXTERNAL_SUGGESTED = {
    "2024-01144": ("transferred_federal", "From your note: functions moved to State. Add URL."),
    "2024-01168": ("replaced", "From your note: DEEP concluded, followed by GANNET. Add URL."),
    "2024-01077": ("transferred_out", "From your note: moved to India's national TB programme. "
                   "Add URL."),
    "2024-01937": ("sunset", "From your note: OCDETF program sunset. Add URL."),
    "2024-01488": ("continued", "From your note: FY2025 Exiger contract (USAspending). Add URL."),
    "2024-00848": ("expanded", "From your note: NLM-Gene-2. Add URL."),
    "2024-00713": ("unknown", HHS_PAGE_NOTE),
    "2024-00830": ("unknown", HHS_PAGE_NOTE),
    "2024-00636": ("unknown", HHS_PAGE_NOTE),
}


def main() -> None:
    wb = load_workbook(SRC)
    ws = wb.active
    col = {ws.cell(HEADER_ROW, j).value: j for j in range(1, ws.max_column + 1)
           if ws.cell(HEADER_ROW, j).value}
    for name in ("external_status", "source_url", "review_note"):
        if name not in col:
            j = max(col.values()) + 1
            ws.cell(HEADER_ROW, j, name).font = Font(bold=True)
            col[name] = j
    for name, w in (("external_status", 20), ("source_url", 40), ("review_note", 70)):
        ws.column_dimensions[ws.cell(HEADER_ROW, col[name]).column_letter].width = w

    dv_rel = DataValidation(type="list", formula1=f'"{RELATIONS}"', allow_blank=True)
    dv_ext = DataValidation(type="list", formula1=f'"{EXTERNAL}"', allow_blank=True)
    ws.add_data_validation(dv_rel)
    ws.add_data_validation(dv_ext)

    input_fill = PatternFill("solid", fgColor="FFF2CC")
    note_fill = PatternFill("solid", fgColor="FCE4D6")
    wrap = Alignment(wrap_text=True, vertical="top")
    seen, n_changed = set(), 0
    for r in range(HEADER_ROW + 1, ws.max_row + 1):
        if ws.cell(r, col["role"]).value != "2024":
            continue
        uid = ws.cell(r, col["record_uid"]).value
        seen.add(uid)
        notes = []
        if uid in CHANGES:
            ids, rel, conf, why = CHANGES[uid]
            before = (f"{ws.cell(r, col['match_cand_ids']).value or ''} | "
                      f"{ws.cell(r, col['relation']).value or ''}")
            ws.cell(r, col["match_cand_ids"], ids)
            ws.cell(r, col["relation"], rel)
            if conf:
                ws.cell(r, col["confidence"], conf)
            notes.append(f"CHANGED (was: {before}). {why}")
            n_changed += 1
        if uid in FLAGS:
            notes.append(FLAGS[uid])
        if uid in EXTERNAL_SUGGESTED:
            status, why = EXTERNAL_SUGGESTED[uid]
            ws.cell(r, col["external_status"], status)
            notes.append(f"external_status suggested: {why}")
        for name in ("external_status", "source_url"):
            ws.cell(r, col[name]).fill = input_fill
        dv_rel.add(ws.cell(r, col["relation"]))
        dv_ext.add(ws.cell(r, col["external_status"]))
        if notes:
            c = ws.cell(r, col["review_note"], " | ".join(notes))
            c.alignment = wrap
            c.fill = note_fill

    missing = (set(CHANGES) | set(FLAGS) | set(EXTERNAL_SUGGESTED)) - seen
    assert not missing, f"record ids not found in sheet: {missing}"
    wb.save(DST)
    print(f"changed {n_changed} records, flagged {len(FLAGS)}, "
          f"external suggestions {len(EXTERNAL_SUGGESTED)} -> {DST}")


if __name__ == "__main__":
    main()