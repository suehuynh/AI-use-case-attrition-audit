# Labeling Codebook

Used for the pilot (development set) and the held-out validation sample. Coders label independently
and never see model output.

## The question

For each dropped 2024 record: **can the same system still be identified in the 2025 inventory?**
Judge from the inventory text in the sheet only (name, agency, bureau, purpose). No web search.

You see one 2024 record and 10 candidate 2025 entries in random order. Choose one relation and list
the matching candidate(s).

## Two relations

| Relation | Use when | Candidates to list |
| --- | --- | --- |
| `SAME_SYSTEM` | The 2024 system is still listed in 2025: in full (same or new name), as part of a broader entry, or split into narrower entries | Every matching entry (one, or all pieces of a split) |
| `NO_MATCH` | The system cannot be identified in 2025 | None |

Also fill `confidence` (high / medium / low) and, when unsure, a short note.

## Decision tree

```
Q1  Does any candidate (or a 2025 entry you found in the 2025_inventory tab) describe the SAME
    system or project, or contain the 2024 entry's function?
    │
    ├─ NO ───────────────────────────────────────────────────────────► NO_MATCH
    │
    └─ YES ─► Q2  Same agency?  (Different bureau of the same department: the 2025 entry must
               │      name the same program, or be department-wide.)
               │
               ├─ NO, and no text showing the program moved ─────────► NO_MATCH
               │
               └─ YES, or the text shows the program moved ──────────► SAME_SYSTEM
                                                                      list every matching entry
```

## Checks at any step

1. **Same product is not same system.** Westlaw at SEC is not Westlaw at DOJ: `NO_MATCH`.
2. **Same task is not same system.** Two agencies' "public comment analysis" tools are different
   systems unless the text shows a transfer: `NO_MATCH`.
3. **Different bureau, same department:** `SAME_SYSTEM` only if the text shows the same specific
   program (matching name, a "previously named" note pointing to it, or the same vendor and
   function), or the 2025 entry is department-wide.
4. **"Previously named X":** check whether X already existed in 2024 at that agency
   (the `2024_inventory` tab of your sheet). If it did, the note belongs to that entry.
5. **A 2025 entry marked Retired still counts** as `SAME_SYSTEM` if it is the same system.
6. **No purpose text:** `SAME_SYSTEM` only with a specific link (name, vendor, rename note);
   otherwise `NO_MATCH` with confidence medium or low.
7. **Two 2024 records matching one 2025 entry** is fine: label each `SAME_SYSTEM`.
8. **Unsure:** pick the better label, set confidence to low, add a note.
9. **No candidate matches:** spend up to 2 minutes searching the `2025_inventory` tab; if you find
   the system there, put its uid (e.g. `2025-01234`) in `found_2025_uid` and choose `SAME_SYSTEM`.

## How human and model answers line up

The model judges each pair separately: `SAME_SYSTEM` or `DIFFERENT`. A coder's `NO_MATCH` for a
record equals the model saying `DIFFERENT` for all 10 pairs; a coder's listed candidates are the
pairs the model should call `SAME_SYSTEM`.

Coders only choose the relation and list candidates. Final record labels (renamed, consolidated,
split, and so on) are computed afterwards by a script from those answers; see the README.
