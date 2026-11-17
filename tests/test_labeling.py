"""Labeling sheets carry their own key: uids are read from the sheet, never matched by position."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.labeling import (HEADER_ROW, PILOT_TITLE, LabeledSheet,  # noqa: E402
                                LabelingSheetBuilder, save_guarded)
from attrition.scoring import PilotScorer  # noqa: E402

PREV = pd.DataFrame([
    {"uid": "2024-1", "agency": "DOJ", "bureau": "", "name": "Westlaw", "purpose": "research"},
    {"uid": "2024-2", "agency": "EPA", "bureau": "", "name": "ServiceNow AI", "purpose": "helpdesk"},
    {"uid": "2024-3", "agency": "HHS", "bureau": "", "name": "Gone tool", "purpose": "x"},
])
NXT = pd.DataFrame([{"uid": f"2025-{i}", "agency": a, "bureau": "", "name": f"Entry {i}", "purpose": "p"}
                    for i, a in enumerate(["DOJ", "SEC", "EPA", "EPA", "DOE", "HHS"])])
CANDS = pd.DataFrame([(p, n, k + 1) for p in PREV["uid"] for k, n in enumerate(NXT["uid"])],
                     columns=["uid_prev", "uid_next", "rank"])


def build(tmp_path, answers):
    """Build a sheet, fill answers {record_uid: (relation, [2025 uids], found)}, save it."""
    wb, pairs = LabelingSheetBuilder(NXT, PREV, seed=3).build(PREV, CANDS, PILOT_TITLE)
    path = tmp_path / "sheet.xlsx"
    wb.save(path)
    wb = load_workbook(path)
    ws = wb["labels"]
    col = {ws.cell(HEADER_ROW, j).value: j for j in range(1, ws.max_column + 1)}
    for r in range(HEADER_ROW + 1, ws.max_row + 1):
        uid = ws.cell(r, col["record_uid"]).value
        if ws.cell(r, col["role"]).value != "2024" or uid not in answers:
            continue
        rel, uids, found = answers[uid]
        ids = pairs[(pairs["record_uid"] == uid) & pairs["uid_next"].isin(uids)]["cand_id"]
        ws.cell(r, col["relation"], rel)
        ws.cell(r, col["match_cand_ids"], ", ".join(ids))
        ws.cell(r, col["confidence"], "high")
        ws.cell(r, col["found_2025_uid"], found)
    wb.save(path)
    return path, pairs


ANSWERS = {"2024-1": ("SAME_SYSTEM", ["2025-0"], ""),
           "2024-2": ("SAME_SYSTEM", ["2025-2", "2025-3"], ""),
           "2024-3": ("NO_MATCH", [], "")}


def model_run(pairs):
    """A model that matches 2024-1 correctly, only one piece of 2024-2, and a false match for 2024-3."""
    hits = {("2024-1", "2025-0"), ("2024-2", "2025-3"), ("2024-3", "2025-5")}
    a = pairs.rename(columns={"record_uid": "uid_prev"}).copy()
    a["relation"] = ["SAME_SYSTEM" if (p, n) in hits else "DIFFERENT" for p, n in zip(a["uid_prev"], a["uid_next"])]
    a["confidence"], a["rationale"] = "high", ""
    return a


def test_sheet_roundtrip_and_hidden_uid(tmp_path):
    path, pairs = build(tmp_path, ANSWERS)
    ws = load_workbook(path)["labels"]
    hidden = [c for c, d in ws.column_dimensions.items() if d.hidden]
    assert len(hidden) == 1 and ws[f"{hidden[0]}{HEADER_ROW}"].value == "uid_next"
    s = LabeledSheet.read(path)
    s.check_pairs(pairs)
    assert s.matched_uids("2024-1") == {"2025-0"}
    assert s.matched_uids("2024-2") == {"2025-2", "2025-3"}
    assert s.matched_uids("2024-3") == set()
    assert {"2025_inventory", "2024_inventory"} <= set(load_workbook(path).sheetnames)


def test_shuffling_rows_does_not_change_scores(tmp_path):
    path, pairs = build(tmp_path, ANSWERS)
    df = pd.read_excel(path, sheet_name="labels", header=HEADER_ROW - 1, dtype=str)
    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    assert not shuffled["record_uid"].equals(df["record_uid"])
    spath = tmp_path / "shuffled.xlsx"
    shuffled.to_excel(spath, sheet_name="labels", startrow=HEADER_ROW - 1, index=False)

    run = model_run(pairs)
    s1, _ = PilotScorer(LabeledSheet.read(path)).score(run)
    s2, _ = PilotScorer(LabeledSheet.read(spath)).score(run)
    assert s1 == s2
    assert s1["records_correct"] == 2 and s1["pair_precision"] == round(2 / 3, 3)


def test_found_uid_counts_as_match(tmp_path):
    path, _ = build(tmp_path, {"2024-3": ("SAME_SYSTEM", [], "2025-99")})
    assert LabeledSheet.read(path).matched_uids("2024-3") == {"2025-99"}


def test_bad_labels_are_reported(tmp_path):
    path, _ = build(tmp_path, {"2024-1": ("PART_OF", ["2025-0"], ""),
                               "2024-2": ("SAME_SYSTEM", [], "")})
    with pytest.raises(ValueError) as e:
        LabeledSheet.read(path)
    assert "PART_OF" in str(e.value) and "no match_cand_ids" in str(e.value)


def test_pairs_mismatch_is_caught(tmp_path):
    path, pairs = build(tmp_path, ANSWERS)
    wrong = pairs.copy()
    wrong.loc[0, "uid_next"] = "2025-77"
    with pytest.raises(ValueError, match="differ"):
        LabeledSheet.read(path).check_pairs(wrong)


def test_run_must_cover_exactly_the_sheet_pairs(tmp_path):
    path, pairs = build(tmp_path, ANSWERS)
    with pytest.raises(ValueError, match="exactly"):
        PilotScorer(LabeledSheet.read(path)).score(model_run(pairs).iloc[:-1])


def test_save_guarded_refuses_overwrite(tmp_path):
    wb, _ = LabelingSheetBuilder(NXT, PREV, seed=3).build(PREV, CANDS, PILOT_TITLE)
    p = tmp_path / "s.xlsx"
    save_guarded(wb, p, force=False)
    with pytest.raises(SystemExit):
        save_guarded(wb, p, force=False)
    save_guarded(wb, p, force=True)
