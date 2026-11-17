"""Pairwise LLM adjudication.

PromptBuilder   reads prompts/adjudicator.txt and fills it for one (2024, 2025) pair
LLMAdjudicator  runs a backend (vllm | transformers | mock) and parses + checks the JSON

The model judges one pair at a time: SAME_SYSTEM or DIFFERENT (docs/codebook.md).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config

RELATIONS = ["SAME_SYSTEM", "DIFFERENT"]
MATCH = "SAME_SYSTEM"
CONFIDENCES = ["high", "medium", "low"]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": RELATIONS},
        "confidence": {"type": "string", "enum": CONFIDENCES},
        "evidence_2024": {"type": "string"},
        "evidence_2025": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["relation", "confidence", "evidence_2024", "evidence_2025", "rationale"],
}


# ---------------------------------------------------------------------------
class PromptBuilder:
    """Splits the prompt file into system text (instructions + few-shot) and a user template."""

    SECTION = re.compile(r"^### (SYSTEM|FEW-SHOT EXAMPLES|USER TEMPLATE)\s*$", re.M)

    def __init__(self, prompt_path: Path, max_field_chars: int):
        text = Path(prompt_path).read_text(encoding="utf-8")
        parts = self.SECTION.split(text)
        sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
        self.system = sections["SYSTEM"] + "\n\n" + sections["FEW-SHOT EXAMPLES"]
        self.user_template = sections["USER TEMPLATE"]
        self.max_chars = max_field_chars

    def _clip(self, value: object) -> str:
        s = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            return "(not reported)"
        return s if len(s) <= self.max_chars else s[: self.max_chars] + " ..."

    @staticmethod
    def merge_system(conv: list[dict[str, str]]) -> list[dict[str, str]]:
        """Some chat templates (Gemma) reject a system role: prepend it to the first user turn."""
        if not conv or conv[0]["role"] != "system":
            return conv
        system, rest = conv[0]["content"], conv[1:]
        first = dict(rest[0])
        first["content"] = system + "\n\n" + first["content"]
        return [first] + rest[1:]

    def messages(self, a: pd.Series, b: pd.Series) -> list[dict[str, str]]:
        fields = {}
        for suffix, rec in (("2024", a), ("2025", b)):
            for f in ("agency", "bureau", "name", "purpose", "outputs"):
                fields[f"{f}_{suffix}"] = self._clip(rec.get(f))
        user = self.user_template.format(**fields)
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
@dataclass
class ParsedOutput:
    valid_json: bool
    relation: str | None
    confidence: str | None
    evidence_2024: str
    evidence_2025: str
    rationale: str
    quote_2024_found: bool
    quote_2025_found: bool


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def parse_output(raw: str, a_text: str, b_text: str) -> ParsedOutput:
    """Parse model JSON and check that each evidence quote appears in its record."""
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    rel = d.get("relation") if d.get("relation") in RELATIONS else None
    conf = d.get("confidence") if d.get("confidence") in CONFIDENCES else None
    e24, e25 = str(d.get("evidence_2024", "")), str(d.get("evidence_2025", ""))
    return ParsedOutput(
        valid_json=rel is not None and conf is not None,
        relation=rel,
        confidence=conf,
        evidence_2024=e24,
        evidence_2025=e25,
        rationale=str(d.get("rationale", "")),
        quote_2024_found=bool(e24) and _norm(e24) in _norm(a_text),
        quote_2025_found=bool(e25) and _norm(e25) in _norm(b_text),
    )


# ---------------------------------------------------------------------------
class LLMAdjudicator:
    def __init__(self, config: Config, model: str, backend: str = "vllm"):
        p = config.raw["adjudication"]
        self.model = model
        self.backend = backend
        self.temperature = p["temperature"]
        self.max_new_tokens = p["max_new_tokens"]
        self.guided_json = p["guided_json"]
        self.seed = p["seed"]
        self.max_model_len = p.get("max_model_len", 4096)
        self.gpu_memory_utilization = p.get("gpu_memory_utilization", 0.90)
        self.prompts = PromptBuilder(config.root / p["prompt_file"], p["max_field_chars"])
        self.merge_system = p.get("merge_system_prompt", "gemma" in model.lower())
        self._engine = None
        self.backend_version = "n/a"

    # ---- backends -----------------------------------------------------
    def _generate_vllm(self, convs: list[list[dict]]) -> list[str]:
        import vllm
        from vllm import LLM, SamplingParams

        self.backend_version = f"vllm=={vllm.__version__}"
        if self._engine is None:
            # Prefix caching reuses the shared system prompt across all calls.
            # max_model_len: Llama-3.1 defaults to 128k context, whose KV cache does not fit
            # on a 24 GB GPU. Our prompts are ~2k tokens, so 4096 is plenty.
            self._engine = LLM(model=self.model, seed=self.seed, enable_prefix_caching=True,
                               max_model_len=self.max_model_len,
                               gpu_memory_utilization=self.gpu_memory_utilization)
        kwargs = dict(temperature=self.temperature, max_tokens=self.max_new_tokens, seed=self.seed)
        if self.guided_json:
            kwargs.update(self._vllm_structured_kwargs())
        outs = self._engine.chat(convs, SamplingParams(**kwargs), use_tqdm=True)
        return [o.outputs[0].text for o in outs]

    @staticmethod
    def _vllm_structured_kwargs() -> dict:
        """The structured-output API was renamed across vLLM releases; try new then old."""
        try:
            from vllm.sampling_params import StructuredOutputsParams
            return {"structured_outputs": StructuredOutputsParams(json=OUTPUT_SCHEMA)}
        except ImportError:
            pass
        try:
            from vllm.sampling_params import GuidedDecodingParams
            return {"guided_decoding": GuidedDecodingParams(json=OUTPUT_SCHEMA)}
        except ImportError:
            print("WARNING: no structured-output API found; relying on prompt + parser.")
            return {}

    def _generate_transformers(self, convs: list[list[dict]]) -> list[str]:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.backend_version = f"transformers=={transformers.__version__}"
        if self._engine is None:
            tok = AutoTokenizer.from_pretrained(self.model)
            mdl = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype="auto", device_map="auto")
            self._engine = (tok, mdl)
        tok, mdl = self._engine
        torch.manual_seed(self.seed)
        texts = []
        for conv in convs:
            ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
            out = mdl.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            texts.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        return texts

    MOCK_LINE = re.compile(r"^(2024|2025) \| Agency: (.*?) \| Bureau: .*? \| Name: (.*)$", re.M)

    def _generate_mock(self, convs: list[list[dict]]) -> list[str]:
        """No model: lets you test the whole pipeline on a laptop. Deterministic rule:
        SAME_SYSTEM when the agencies match and the names are near-identical, else DIFFERENT."""
        from rapidfuzz import fuzz

        self.backend_version = "mock"
        outs = []
        for conv in convs:
            # the pair is in the last two record lines (few-shot examples come earlier)
            found = {y: (ag, nm) for y, ag, nm in self.MOCK_LINE.findall(conv[-1]["content"])}
            (ag_a, nm_a), (ag_b, nm_b) = found.get("2024", ("", "")), found.get("2025", ("", ""))
            same = ag_a == ag_b and fuzz.token_set_ratio(nm_a.lower(), nm_b.lower()) >= 90
            outs.append(json.dumps({"relation": "SAME_SYSTEM" if same else "DIFFERENT",
                                    "confidence": "medium", "evidence_2024": "", "evidence_2025": "",
                                    "rationale": "mock: agency + name rule"}))
        return outs

    def generate(self, convs: list[list[dict]]) -> list[str]:
        if self.backend == "vllm":
            return self._generate_vllm(convs)
        if self.backend == "transformers":
            return self._generate_transformers(convs)
        if self.backend == "mock":
            return self._generate_mock(convs)
        raise ValueError(f"unknown backend {self.backend}")

    # ---- public -------------------------------------------------------
    def adjudicate(self, pairs: pd.DataFrame, prev: pd.DataFrame, nxt: pd.DataFrame) -> pd.DataFrame:
        """pairs needs uid_prev, uid_next, rank. Returns one row per pair with parsed output."""
        P = prev.set_index("uid")
        N = nxt.set_index("uid")
        convs, texts = [], []
        for r in pairs.itertuples():
            a, b = P.loc[r.uid_prev], N.loc[r.uid_next]
            conv = self.prompts.messages(a, b)
            convs.append(self.prompts.merge_system(conv) if self.merge_system else conv)
            texts.append((" ".join(str(a.get(f, "")) for f in ("name", "purpose", "outputs")),
                          " ".join(str(b.get(f, "")) for f in ("name", "purpose", "outputs"))))
        raws = self.generate(convs)
        rows = []
        for r, raw, (ta, tb) in zip(pairs.itertuples(), raws, texts):
            p = parse_output(raw, ta, tb)
            rows.append({"uid_prev": r.uid_prev, "uid_next": r.uid_next, "rank": r.rank,
                         "model": self.model, "backend": self.backend_version,
                         **p.__dict__, "raw": raw})
        return pd.DataFrame(rows)


def run_name(stage: str, model: str) -> str:
    """Adjudication run name: <stage>__<model>, e.g. pilot__Qwen2.5-7B-Instruct."""
    if stage not in ("pilot", "full"):
        raise ValueError(f"stage must be pilot or full, not {stage!r}")
    return f"{stage}__{model.split('/')[-1]}"
