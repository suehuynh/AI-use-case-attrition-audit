import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.sampling import PilotSampler, ValidationSampler  # noqa: E402
from attrition.validation import (cohen_kappa, gold_class, stratified_total,  # noqa: E402
                                  weighted_precision_recall)


def labels():
    rows = []
    for i in range(60):
        rows.append({"uid": f"A{i}", "final_label": "absent", "prior_explanation": "none"})
    for i in range(30):
        rows.append({"uid": f"R{i}", "final_label": "renamed", "prior_explanation": "none"})
    for i in range(5):
        rows.append({"uid": f"T{i}", "final_label": "renamed", "prior_explanation": "retired_in_2024"})
    rows.append({"uid": "K0", "final_label": "moved_to_cots", "prior_explanation": "none"})
    return pd.DataFrame(rows)


def test_sampler_strata_weights_and_exclusion():
    s = ValidationSampler({"absent": 10, "renamed": 6, "retired_matched": 10}, seed=1)
    m = s.sample(labels(), exclude={"A0"})
    assert set(m["stratum"]) == {"absent", "renamed", "retired_matched"}  # cots not sampled
    assert "A0" not in set(m["uid"])
    assert m.groupby("stratum")["n_h"].first().to_dict() == {"absent": 10, "renamed": 6,
                                                             "retired_matched": 5}
    assert m.loc[m["stratum"] == "absent", "weight"].iloc[0] == 59 / 10


def test_stratified_total_exact_when_census():
    df = pd.DataFrame({"stratum": ["a"] * 4 + ["b"] * 2, "N_h": [4] * 4 + [2] * 2,
                       "y": [1, 0, 1, 0, 1, 1]})
    total, se = stratified_total(df, "y")
    assert total == 4 and se == 0  # full census -> no sampling error


def test_gold_class():
    key = pd.DataFrame({"cand_id": ["C1", "C2"], "uid_next": ["N1", "N2"]})
    ag = pd.Series({"N1": "DOJ", "N2": "SEC", "N9": "DOJ"})
    assert gold_class("SAME_SYSTEM", "C1", "", "DOJ", key, ag) == "same_agency"
    assert gold_class("SAME_SYSTEM", "C2", "", "DOJ", key, ag) == "cross_agency"
    assert gold_class("NO_MATCH", "C1", "", "DOJ", key, ag) == "absent"  # ids ignored
    assert gold_class("SAME_SYSTEM", "", "N9", "DOJ", key, ag) == "same_agency"  # found by search


def test_kappa_and_weighted_pr():
    a = pd.Series(["x", "x", "y", "y"])
    assert cohen_kappa(a, a) == 1
    df = pd.DataFrame({"pred": ["s", "s", "a"], "gold": ["s", "a", "a"], "weight": [1, 3, 1]})
    p, r = weighted_precision_recall(df, "pred", "gold", "s")
    assert p == 1 / 4 and r == 1


def test_pilot_sampler_is_deterministic_and_covers_explanations():
    d = labels()
    a, b = PilotSampler.sample(d, 10, seed=0), PilotSampler.sample(d, 10, seed=0)
    assert a["uid"].tolist() == b["uid"].tolist() and len(a) == 10
    assert set(a["prior_explanation"]) == {"none", "retired_in_2024"}
