import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attrition.baseline import BaselineMatcher  # noqa: E402
from attrition.config import Config  # noqa: E402
from attrition.schema import normalize_text  # noqa: E402


def frame(year, rows):
    df = pd.DataFrame(rows, columns=["agency", "name", "purpose"])
    df["name_key"] = df["name"].map(normalize_text)
    df["uid"] = [f"{year}-{i}" for i in range(len(df))]
    return df


@pytest.fixture
def matcher():
    return BaselineMatcher(Config.load(ROOT / "config.yaml"))


def test_exact_match_ignores_case_and_whitespace(matcher):
    prev = frame("24", [("VA", "Opioid  Risk Tool", "x")])
    nxt = frame("25", [("VA", "opioid risk tool", "y")])
    s = matcher.match(prev, nxt).summary()
    assert s["exact"] == 1 and s["dropped"] == 0


def test_no_cross_agency_matches(matcher):
    prev = frame("24", [("VA", "Opioid Risk Tool", "x")])
    nxt = frame("25", [("HHS", "Opioid Risk Tool", "x")])
    assert matcher.match(prev, nxt).summary()["dropped"] == 1


def test_fuzzy_match_on_renamed_record(matcher):
    prev = frame("24", [("DOI", "Wildfire smoke forecasting model", "predict smoke plumes")])
    nxt = frame("25", [("DOI", "Wildfire smoke forecast model v2", "predict smoke plumes")])
    s = matcher.match(prev, nxt).summary()
    assert s["fuzzy"] == 1


def test_name_only_fallback_threshold(matcher):
    prev = frame("24", [("DOI", "Wildfire smoke forecasting model", "")])
    nxt = frame("25", [("DOI", "Payroll chatbot", "")])
    assert matcher.match(prev, nxt).summary()["persisted"] == 0


def test_one_to_one_greedy(matcher):
    prev = frame("24", [("DOE", "Grid load forecaster", "p"), ("DOE", "Grid load forecaster", "p")])
    nxt = frame("25", [("DOE", "Grid load forecaster", "p")])
    s = matcher.match(prev, nxt).summary()
    assert s["persisted"] == 1 and s["dropped"] == 1