# Labeling Codebook (v2)

Used for the pilot and for the Phase 2 validation sample. Two coders label independently.

## Two separate questions

| | Question | Evidence allowed | Used for |
|---|---|---|---|
| **Q1. Linkage** | Does this 2024 entry appear in the 2025 inventory? | The inventory text shown in the sheet only | Scoring the LLM; attrition counts |
| **Q2. Reality** | Does the system still operate outside the inventory? | Primary external sources, with a URL | Reporting-gap analysis of absent records |

Q1 must not use outside knowledge: the LLM only sees inventory text, so a Q1 label based on outside
facts would grade the model on information it never had. Outside findings go in Q2.

## Q1: relation (one per record)

| Relation | Meaning | Candidates listed |
|---|---|---|
| `SAME_SYSTEM` | Same deployed system or project. Name may change (rename, version, ID tag); agency may change only if the text shows the same specific program moved. | Usually one. More than one only if the 2025 inventory lists the same system twice. |
| `PART_OF` | The 2024 entry was absorbed into a broader 2025 entry. | One (the broader entry). |
| `SPLIT_INTO` | A broad 2024 entry was divided into several specific 2025 entries. | Two or more, same agency. |
| `SAME_PRODUCT_DIFFERENT_DEPLOYMENT` | Best candidate uses the same commercial product but is a separate deployment (usually another agency). Counts as no match. | The lookalike, for the record only. |
| `NO_MATCH` | None of the 10 candidates is the same system. | None. |

Rules
1. **Same product is not same system.** Westlaw at SEC is not Westlaw at DOJ. List only the
   same-agency entry as the match; other agencies' copies are `SAME_PRODUCT_DIFFERENT_DEPLOYMENT`.
2. **Same task is not same system.** Two agencies' "public comment analysis" tools are different
   systems unless the text shows a transfer.
3. **One label per record.** If unsure, choose the best label and set confidence to `low`;
   explain in notes.
4. **Two 2024 records matching one 2025 entry** is fine: label each `SAME_SYSTEM`; the aggregation
   rules turn that into `consolidated`.

## Q2: external_status (only for `NO_MATCH` records)

| Code | Meaning |
|---|---|
| `continued` | Still operating at the same agency (e.g. active contract, current agency page) |
| `expanded` | Still operating, scope grew |
| `replaced` | Ended and a successor system took over |
| `transferred_federal` | Moved to another federal agency |
| `transferred_out` | Moved outside the federal government (e.g. a partner government) |
| `in_agency_inventory` | Listed in the agency's own **2025** inventory but not in the OMB file |
| `paused` | Temporarily halted |
| `sunset` | Ended with no successor |
| `unknown` | No reliable evidence found |

Source rules
- Every code except `unknown` needs a **primary** source in `source_url`: agency page,
  USAspending/SAM.gov record, Federal Register, official report. No URL means the code is not used.
- Do not rely on search-engine or AI summaries; open the source and confirm the claim.
- Agency inventory pages: confirm the **reporting year**. HealthIT.gov pages with `HHS-CMS-000xx`
  IDs use the 2024 format (rights/safety impact, 2024 dates); HHS's 2025 inventory has no unique IDs.
  A recent "page updated" date is not evidence of 2025 reporting.

## Final record labels (derived by rules, not hand-coded)

`withheld` > `renamed` (SAME_SYSTEM, same agency) > `cross_agency_move` (SAME_SYSTEM, other agency)
> `consolidated` (PART_OF, or 2+ records to one entry) > `split` (SPLIT_INTO) > `moved_to_cots`
> `absent`.
