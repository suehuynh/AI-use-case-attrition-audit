import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attrition.adjudicator import PromptBuilder, parse_output  # noqa: E402
from attrition.config import Config  # noqa: E402


def builder():
    cfg = Config.load(ROOT / "config.yaml")
    p = cfg.raw["adjudication"]
    return PromptBuilder(ROOT / p["prompt_file"], p["max_field_chars"])


def test_prompt_fills_all_fields():
    a = pd.Series({"agency": "VA", "bureau": "VHA", "name": "Opioid risk tool",
                   "purpose": "flag risk", "outputs": ""})
    b = pd.Series({"agency": "VA", "bureau": "VHA", "name": "STORM", "purpose": "flag risk",
                   "outputs": "scores"})
    msgs = builder().messages(a, b)
    assert msgs[0]["role"] == "system" and "SAME_SYSTEM" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "{" not in user and "Opioid risk tool" in user and "(not reported)" in user


def test_parse_valid_and_quote_check():
    raw = ('{"relation": "SAME_SYSTEM", "confidence": "high", "evidence_2024": "flag risk",'
           ' "evidence_2025": "made up quote", "rationale": "x"}')
    p = parse_output(raw, "Opioid tool flag risk", "STORM flag risk")
    assert p.valid_json and p.relation == "SAME_SYSTEM"
    assert p.quote_2024_found and not p.quote_2025_found


def test_parse_invalid_relation():
    p = parse_output('{"relation": "MAYBE", "confidence": "high"}', "", "")
    assert not p.valid_json and p.relation is None


def test_parse_garbage():
    assert not parse_output("I think they are the same.", "", "").valid_json


def test_v3_prompt_declares_two_relations():
    cfg = Config.load(ROOT / "config.yaml")
    b = PromptBuilder(ROOT / "prompts/adjudicator_v3.txt", cfg.raw["adjudication"]["max_field_chars"])
    assert b.relations == ["SAME_SYSTEM", "DIFFERENT"]
    assert "RELATIONS" not in b.system and "SAME_SYSTEM" in b.system


def test_old_prompt_keeps_full_relation_set():
    b = PromptBuilder(ROOT / "prompts/adjudicator_v1.txt", 600)
    assert "PART_OF" in b.relations


def test_parse_rejects_relation_outside_prompt():
    raw = '{"relation": "PART_OF", "confidence": "high", "evidence_2024": "", "evidence_2025": "", "rationale": ""}'
    assert not parse_output(raw, "", "", ["SAME_SYSTEM", "DIFFERENT"]).valid_json
    assert parse_output(raw.replace("PART_OF", "DIFFERENT"), "", "", ["SAME_SYSTEM", "DIFFERENT"]).valid_json


def test_merge_system_for_gemma():
    conv = [{"role": "system", "content": "RULES"}, {"role": "user", "content": "Compare"}]
    merged = PromptBuilder.merge_system(conv)
    assert [m["role"] for m in merged] == ["user"]
    assert merged[0]["content"].startswith("RULES") and merged[0]["content"].endswith("Compare")
