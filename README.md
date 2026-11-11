# Attrition Audit: Federal AI Use Case Inventory (2024 → 2025)

Audits why AI use cases in the 2024 federal inventory are absent from 2025, and how much of the
measured attrition is linkage error. Phase 1 step 1 replicates the year-over-year matcher of
Lurie et al. (2026), "Triangulating Across U.S. Federal AI Transparency Regimes", Appendix A.

## Data
Official OMB releases, fetched from GitHub and copied into `data/raw/` (run from the project root):

```bash
git clone --depth 1 https://github.com/ombegov/2024-Federal-AI-Use-Case-Inventory.git inv2024
git clone --depth 1 https://github.com/ombegov/2025-Federal-Agency-AI-Use-Case-Inventory.git inv2025

cp inv2024/data/2024_consolidated_ai_inventory_raw_v2.csv data/raw/
cp inv2025/Data/2025_individually_reported_AI_use_cases.csv data/raw/
```

Usage notes:
- **2024 file:** use `_raw_v2.csv` (2,133 rows), the corrected re-release. The original `_raw.csv`
  has 1,757 rows and numbered column names (`2_use_case_name`, ...), so it does not match the
  column map in `config.yaml`. The file is Windows-1252 encoded (`encoding: cp1252` in config).
- **2025 file:** use only the individually reported file (3,611 rows). The separate
  `2025_consolidated_COTS_AI_use_cases.csv` (900 rows of commercial off-the-shelf products) is
  excluded; after deduplication this gives the paper's 3,542 total.
- **Pinned versions:** the clones above pull the latest commit. Results in this README were
  produced from commits `4a29d13` (2024 repo) and `06c7ebe` (2025 repo). If OMB updates either
  repo, check out these commits to reproduce the exact inputs (fetching by commit needs the full hash):
  ```bash
  git -C inv2024 fetch --depth 1 origin 4a29d132291261d4829b187acd8af618e7295294 && git -C inv2024 checkout FETCH_HEAD
  git -C inv2025 fetch --depth 1 origin 06c7ebeef5b376524211042bd9673b2a7fefd3e3 && git -C inv2025 checkout FETCH_HEAD
  ```

## Run
```
pip install pandas rapidfuzz pyyaml pytest
python scripts/run_baseline.py
pytest -q
```
All parameters live in `config.yaml`.

## Layout
```
config.yaml              single source of truth
src/attrition/
  config.py              Config
  schema.py              SchemaNormalizer (per-year column maps -> canonical schema)
  loader.py              InventoryLoader (read, normalize, filter)
  baseline.py            BaselineMatcher (exact + within-agency fuzzy, greedy 1:1)
  retrieval.py           CandidateRetriever (TF-IDF + fuzzy name + agency bonus, top-k, all agencies)
scripts/run_baseline.py  replication run, writes outputs/
scripts/run_retrieval.py tags explained drops, retrieves top-10 2025 candidates per drop
tests/                   unit tests for matcher behaviour
```

## Replication status
| | Paper (main text) | Ours |
|---|---|---|
| 2024 records | 2,050 | 2,050 |
| 2025 records | 3,474 used / 3,542 total | 3,542 |
| Persisted | 1,227 | 1,224 |
| Dropped | 823 | 826 |
| New | 2,247 | 2,318 |

Settings that reproduce the paper: dedupe on (name, agency); 2025 purpose = `problem_solved` only;
case-folded exact key. Remaining gap: 3 records. The 2025 "new" count still differs by ~70,
suggesting the paper dropped additional 2025 rows after the totals step (open question for authors).

Sensitivity: defensible preprocessing choices alone move the dropped count from 802 to 850.

## Phase 1 step 2: candidate retrieval
`python scripts/run_retrieval.py` (about 15 seconds). Score = 0.5 TF-IDF cosine (name x2 + purpose)
+ 0.3 fuzzy name (token_set_ratio) + 0.2 same-agency bonus; top 10 per dropped record from all
3,542 2025 records, including ones the baseline already matched (needed to detect consolidation).

| Check | Result |
|---|---|
| Recall@10 on the 1,224 baseline-matched pairs | 100% (recall@1: 98.8% exact, 95.2% fuzzy) |
| Candidate pairs for the 826 drops | 8,260 |
| Drops with a prior explanation | 220 (USAID dissolved: 137; retired in 2024: 116; both: 33) |
| Unexplained drops | 606 |

Agency-weight sensitivity (share of top-1 candidates from a different agency):
0.0 → 66.8%, 0.1 → 42.1%, **0.2 → 30.1%**, 0.4 → 25.5%. The bonus does not act as a filter at 0.2.
Among the 606 unexplained drops, 17.2% of top-1 candidates are cross-agency.

### Linkage-error findings so far
- **ID-tag artifact (31 records).** 125 DOI records in 2025 append `[2024 INV#WO...]` to the use
  case name. Stripping that tag before baseline matching recovers 31 matches (dropped 826 → 795).
  These are renames by formatting only.
- **Moved to the COTS checklist (≥9 records).** 2025 split commercial products into a separate
  checklist (`2025_consolidated_COTS_AI_use_cases.csv`, 45 agencies x 20 tasks). Nine dropped records
  match a same-agency 2025 COTS product name (e.g. VA Github Copilot, EAC Microsoft Copilot).
  This is a lower bound: name matching only, and 16 of 45 COTS agencies are not yet mapped to
  abbreviations. Saved to `outputs/cots_candidates.csv`.
- **Generic commercial tools dominate cross-agency candidates** (Copilot, Adobe, Visual Studio).
  These are the same product at different agencies, not moved systems. The adjudicator prompt must
  separate "same product, different deployment" from "same system".