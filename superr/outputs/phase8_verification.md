# Phase 8 — Final verification

Run on 2026-11-17 after the methodology report was written.

## What was run

1. **The fresh `python -m src.pipeline --force` run (Phase 6 close):** reproduced all 100 compared output tables and ball files exactly against the pre-run backup.
2. **Since then, a cached run of the same command** (models reused from `outputs/cache/`, everything from Phase 2b tables onwards rebuilt), with the Phase 7 code: label column, report figures and cross-check. Exit code 0, no errors or tracebacks.
3. **`pytest`:** 63 tests pass (58 before Phase 7, plus 5 for the labels and the cross-check mechanics).
4. **`python -m src.report.check_report`:** 196 claims checked, 196 pass, 0 mismatches, 0 missing. 72 numbers in the report are not covered by an automatic check; all are listed in `outputs/report/crosscheck.md`. They are definitions (LI 1.0, top 25%), worked scenario inputs ("need 60 off 12"), unit labels, and a few figures quoted from earlier phases' logs (774 training matches, 817 distinct values in the leak, 100 compared files).
5. **Phase 8 checks on the finished outputs** (all pass):

| Check | Result |
|---|---|
| `final_rankings.csv` has role, rank, player_id, player_name, teams, label, note, evidence | Pass |
| 10 batters and 10 bowlers, ranked 1–10, no missing ids, names or teams | Pass |
| Rashid Khan is id 4722 wherever he appears | Pass |
| Labels are only "All-round" / "Pressure-specific"; short-record note only for Sooryavanshi and Inglis | Pass |
| Final rankings identical to before the label change, apart from `interpretation` → `label` + `note` (23 columns compared) | Pass |
| All 13 figures linked in the report exist, are 300 dpi, and each has a numbered takeaway caption; no unused figure files | Pass |
| The report has every required section (0–6, Appendices A–E) and no template placeholders | Pass |
| Every pinned package in `requirements.txt` matches the installed version | Pass |

## Length

Rendered to A4 at 10 pt with 18–20 mm margins, the main body (§0–§6) is **14 pages** and the appendices 5 more. Other renderers will paginate differently.

## Changes made while finishing (all recorded in DECISIONS.md D38–D39)

- **Per-player disclaimer replaced by a `label` column** (Pressure-specific when option B rank > 30; All-round otherwise), with the caveat stated once in the report (§5.3). The short-record flag is a separate `note` column.
- **Statements corrected against the current tables:** the eligibility-200 variant removes two batters below the bar (not three) and Klaasen slips to 11th; excluding associates keeps 8 (not 9–10) of the bowling top 10; the uplift sample range is 100–888 pressure balls; India's and KKR's batting pressure shares are 17.1% and 18.9%.
- **The cached-run time varies with machine load** (4 to 11 minutes seen), so the report says "under 15 minutes" and the check verifies that bound.

## Still open

- **The Unstop dropdown spellings** have not been received, so the names have not been checked against them. Names are the official mapping-file spellings and every row carries `player_id`.

# Phase 9 — Submission PDF and repository

## Content fixes (D40)

- **§4.4 tables** now show Rating (the composite that sets the order) and Consistency, and ‡ marks every position set by the tie-break. A generated footnote under each table names the players that could not be told apart, with their option A and B ranks.
- **Rank pools stated:** §4.5 overall ranks are among the eligible players (137 batters, 139 bowlers), the same pool as the pressure ranking. §3.2 and the Narine insight state their own pool (at least 500 balls).
- **Sooryavanshi card:** the raw +84.8 is broken into runs +76.0, wicket term +1.9, non-striker run-out term +0.7 and opposition adjustment +6.2. The check verifies the four parts sum to the raw value.
- **Author block** added (repository link still `[REPO_LINK]`); the names caveat now uses neutral wording. "Bhopal" appears nowhere in the project.
- **Cross-check:** 202 claims, 0 mismatches, 0 missing.

## The PDF

- **Built by** `python -m src.report.build_pdf` (the last step of `python -m src.pipeline`): Markdown → HTML → headless Edge print.
- **Layout:** title page, clickable contents, bookmarks for every section and subsection, "Page X of Y" footer, running header.
- **Tables and figures:** zebra tables with repeated headers, figures embedded at 200 dpi.
- **Size:** 20 pages, 2.8 MB.
  - Title page and contents: pages 1–2.
  - Main body (§0–§6): pages 3–15, 13 pages.
  - Appendices A–E: pages 16–20, 5 pages.
- **Inspection:** every page was rendered to PNG and checked. Fixed along the way:
  - squeezed 12-column tables;
  - a clipped axis label;
  - doubled bookmark titles;
  - a heatmap that left a gap.
- **Remaining layout:** the gaps left are the end of §6 before the appendices (which start on a new page) and the last page.
- **Text check:** all 168 cross-checked claims appear in the PDF's extracted text. The two that don't match verbatim are extraction artefacts: a hyphen at a line break, and a space after a code span.

## Repository (not pushed)

- **`.gitignore`** excludes `data/raw/`, `.venv/`, caches, `outputs/cache/` and the three ball-level CSVs.
- **Raw data** now lives in `data/raw/` (`config.yaml` → `paths.data_dir`). README section "Getting the data" lists the six file names.
- **Dry run:** a `git add` dry run (through a throwaway git directory outside the project) would add 210 files.
  - No raw CSV, cache, `.venv` or ball-level file.
  - Largest file: the 2.8 MB PDF.
  - Closest thing to raw data: `outputs/tables/phase2b/close_finish_curves.csv`, 971 derived ball rows for the four close finishes.
