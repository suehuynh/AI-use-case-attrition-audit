import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.classifier import AttritionClassifier  # noqa: E402

NXT = pd.DataFrame([
    {"uid": "N1", "agency": "DOJ", "name": "Westlaw AI", "withheld": "No"},
    {"uid": "N2", "agency": "STATE", "name": "Malaria model", "withheld": "No"},
    {"uid": "N3", "agency": "USDA", "name": "Comment analysis", "withheld": "No"},
    {"uid": "N4", "agency": "DOL", "name": "Hidden tool", "withheld": "Yes - Disclosure Risk"},
    {"uid": "N5", "agency": "VA", "name": "Persisting system", "withheld": "No"},
    {"uid": "N6", "agency": "EPA", "name": "Helpdesk chat", "withheld": "No"},
    {"uid": "N7", "agency": "EPA", "name": "Records system", "withheld": "No"},
    {"uid": "N8", "agency": "DOI", "name": "Image generation", "withheld": "No"},
])

DROPPED = pd.DataFrame([
    {"uid": "P1", "agency": "DOJ", "name": "Westlaw"},             # renamed
    {"uid": "P2", "agency": "USAID", "name": "Malaria"},           # cross-agency move
    {"uid": "P3", "agency": "USDA", "name": "OCIO comments"},      # consolidated (shared N3)
    {"uid": "P4", "agency": "USDA", "name": "FS comments"},        # consolidated (shared N3)
    {"uid": "P5", "agency": "DOL", "name": "Tool"},                # withheld
    {"uid": "P6", "agency": "VA", "name": "Old name"},             # consolidated (baseline target)
    {"uid": "P7", "agency": "EPA", "name": "ServiceNow AI"},       # split (two EPA entries)
    {"uid": "P8", "agency": "VA", "name": "Copilot"},              # moved_to_cots
    {"uid": "P9", "agency": "HHS", "name": "Gone"},                # absent
    {"uid": "P10", "agency": "DOI", "name": "Caseguard"},          # renamed (N8 shared only cross-agency)
    {"uid": "P11", "agency": "NCUA", "name": "Image editor"},      # cross-agency, not consolidated
])


def pair(p, n, rel, conf="high", rank=1):
    return {"uid_prev": p, "uid_next": n, "rank": rank, "relation": rel, "confidence": conf}


ADJ = pd.DataFrame([
    pair("P1", "N1", "SAME_SYSTEM"),
    pair("P1", "N2", "DIFFERENT", rank=2),
    pair("P2", "N2", "SAME_SYSTEM"),
    pair("P3", "N3", "SAME_SYSTEM"),
    pair("P4", "N3", "SAME_SYSTEM", "medium"),
    pair("P5", "N4", "SAME_SYSTEM"),
    pair("P6", "N5", "SAME_SYSTEM"),
    pair("P7", "N6", "SAME_SYSTEM"),
    pair("P7", "N7", "SAME_SYSTEM", rank=2),
    pair("P8", "N1", "DIFFERENT"),
    pair("P9", "N1", "DIFFERENT"),
    pair("P10", "N8", "SAME_SYSTEM"),
    pair("P11", "N8", "SAME_SYSTEM"),
])


def labels(min_conf="low"):
    clf = AttritionClassifier(NXT, baseline_targets={"N5"}, cots_uids={"P8"})
    return dict(zip(DROPPED["uid"], clf.classify(DROPPED, ADJ, min_conf)["final_label"]))


def test_all_rules():
    assert labels() == {
        "P1": "renamed", "P2": "cross_agency_move", "P3": "consolidated", "P4": "consolidated",
        "P5": "withheld", "P6": "consolidated", "P7": "split", "P8": "moved_to_cots",
        "P9": "absent", "P10": "renamed", "P11": "cross_agency_move",
    }


def test_high_confidence_only_drops_medium_match():
    lab = labels("high")
    assert lab["P4"] == "absent"      # its only match was medium confidence
    assert lab["P3"] == "renamed"     # N3 no longer shared once P4 is dropped


def test_split_needs_two_same_agency_entries():
    """A second SAME_SYSTEM match at another agency does not make a split."""
    adj = pd.DataFrame([pair("P1", "N1", "SAME_SYSTEM"), pair("P1", "N2", "SAME_SYSTEM", rank=2)])
    d = DROPPED[DROPPED["uid"] == "P1"]
    out = AttritionClassifier(NXT, set(), set()).classify(d, adj).iloc[0]
    assert out["final_label"] == "renamed" and out["n_matches"] == 2


def test_consolidation_reason():
    out = AttritionClassifier(NXT, baseline_targets={"N5"}, cots_uids={"P8"}).classify(DROPPED, ADJ)
    reason = dict(zip(out["uid"], out["consolidation_reason"]))
    assert reason["P3"] == reason["P4"] == "shared_with_dropped"
    assert reason["P6"] == "shared_with_persisting"


def test_ignores_relations_other_than_same_system():
    adj = pd.DataFrame([pair("P1", "N1", "PART_OF")])
    d = DROPPED[DROPPED["uid"] == "P1"]
    assert AttritionClassifier(NXT, set(), set()).classify(d, adj).iloc[0]["final_label"] == "absent"
