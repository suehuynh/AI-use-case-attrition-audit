import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.adjudicator import run_name  # noqa: E402
from attrition.scoring import choose_model  # noqa: E402


def scores(rows):
    return pd.DataFrame(rows, columns=["model", "records_correct", "pair_kappa", "beats_baseline"])


def test_highest_accuracy_among_models_beating_baseline():
    c = choose_model(scores([("A", 40, 0.5, True), ("B", 45, 0.4, True), ("C", 48, 0.9, False)]))
    assert c["chosen"] == "B" and c["eligible"] == ["A", "B"]


def test_kappa_breaks_ties():
    c = choose_model(scores([("A", 45, 0.5, True), ("B", 45, 0.6, True)]))
    assert c["chosen"] == "B" and c["reason"] == "kappa tie-break"


def test_no_model_beats_baseline():
    assert choose_model(scores([("A", 20, 0.1, False)]))["chosen"] is None


def test_run_names():
    assert run_name("pilot", "Qwen/Qwen2.5-7B-Instruct") == "pilot__Qwen2.5-7B-Instruct"
    assert run_name("full", "google/gemma-2-9b-it") == "full__gemma-2-9b-it"
