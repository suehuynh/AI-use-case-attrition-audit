"""Tests for CandidateRetriever. These fail until retrieval.py is implemented."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attrition.config import Config  # noqa: E402
from attrition.retrieval import CandidateRetriever  # noqa: E402
from attrition.schema import normalize_text  # noqa: E402


def frame(year, rows):
    df = pd.DataFrame(rows, columns=["agency", "name", "purpose"])
    df["name_key"] = df["name"].map(normalize_text)
    df["uid"] = [f"{year}-{i}" for i in range(len(df))]
    return df


@pytest.fixture
def retriever():
    return CandidateRetriever(Config.load(ROOT / "config.yaml"))


def test_cross_agency_candidate_is_reachable(retriever):
    # A USAID program reappearing under STATE should rank first.
    prev = frame("24", [("USAID", "Malaria outbreak forecasting", "predict malaria outbreaks in partner countries")])
    nxt = frame("25", [
        ("STATE", "Malaria outbreak forecasting model", "predict malaria outbreaks in partner countries"),
        ("USAID", "Payroll chatbot", "answer HR questions"),
        ("DOE", "Grid load forecaster", "forecast electricity demand"),
    ])
    cands = retriever.retrieve(prev, nxt)
    top = cands[cands["rank"] == 1].iloc[0]
    assert top["uid_next"] == "25-0"


def test_returns_at_most_top_k(retriever):
    prev = frame("24", [("VA", "Opioid risk tool", "flag opioid risk")])
    nxt = frame("25", [("VA", f"Tool {i}", "misc") for i in range(30)])
    cands = retriever.retrieve(prev, nxt)
    assert len(cands) <= retriever.top_k


def test_same_agency_bonus_breaks_ties(retriever):
    prev = frame("24", [("VA", "Opioid risk tool", "flag opioid risk")])
    nxt = frame("25", [("HHS", "Opioid risk tool", "flag opioid risk"),
                       ("VA", "Opioid risk tool", "flag opioid risk")])
    cands = retriever.retrieve(prev, nxt)
    assert cands.sort_values("rank").iloc[0]["uid_next"] == "25-1"