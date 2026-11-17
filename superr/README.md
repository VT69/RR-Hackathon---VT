# SupeRR Selector Season 2 — Round 1

Identify the most reliable batters and bowlers under pressure from IPL and T20I ball-by-ball data (2022 onwards).

## Layout

```
config.yaml        all paths, labels, thresholds, model settings, seeds
src/pipeline.py    one command that runs every stage
src/data/          loading, cricket-correct cleaning, data audit
src/features/      match state, leverage, pressure flags, rule-based index, validation tables
src/models/        outcome model, DP win probability, fitted win-probability models, cross-fitting
src/metrics/       batting and bowling metrics          (Phase 3)
src/rating/        shrinkage, composite, ranking        (Phase 4-5)
src/robustness/    sensitivity, tie-break, stability    (Phase 6)
src/report/        submission files, the report's number cross-check     (Phase 7)
src/viz/           chart style, phase figures and the report figures
tests/             pytest for the cricket logic, match state and the DP solver
outputs/           audit, tables, figures, ball-level pressure, reports
DECISIONS.md       every decision, alternatives and evidence
```

## Getting the data

The competition data is not included in this repository. Place the six CSVs from the SupeRR Selector Season 2 data pack in `data/raw/` with these exact file names:

```
data/raw/ball_by_ball.csv
data/raw/matches.csv
data/raw/competitions.csv
data/raw/teams.csv
data/raw/playing_eleven.csv
data/raw/competition_team_player_mapping.csv
```

The folder is set by `paths.data_dir` in `config.yaml` and is ignored by git.

## Reproduce

Python 3.12. The PDF step needs Microsoft Edge or Google Chrome (headless printing); the browser paths tried are in `config.yaml` (`report.pdf.browsers`).

Create the virtual environment and install the pinned packages (Windows paths shown; use `.venv/bin/python` on macOS/Linux):

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

Run everything from the raw CSVs (about 52 minutes on a 4-core laptop; most of it is solving the win-probability tables):

```bash
.venv/Scripts/python -m src.pipeline --force
```

Without `--force` the two slow model steps are read from `outputs/cache/` and the rest is rebuilt (4–11 minutes, depending on machine load):

```bash
.venv/Scripts/python -m src.pipeline
```

Run the tests:

```bash
.venv/Scripts/python -m pytest tests
```

Re-check every number quoted in the methodology report against the output tables (writes `outputs/report/crosscheck.md`; also run at the end of the pipeline), and rebuild the submission PDF on its own:

```bash
.venv/Scripts/python -m src.report.check_report
```

```bash
.venv/Scripts/python -m src.report.build_pdf
```

## What the pipeline writes

| Output | Content |
|---|---|
| `outputs/audit/` | Data-audit facts and tables behind `outputs/data_audit.md` |
| `outputs/ball_pressure.csv` (not committed; rebuilt) | One row per in-scope ball: match state, win probability, leverage, Leverage Index, pressure flags |
| `outputs/tables/phase2b/` | Model comparison, calibration, leverage distribution, exposure by phase / player / team, separation tests, close finishes |
| `outputs/figures/phase2b/` | The corresponding charts |
| `outputs/ball_value.csv` (not committed; rebuilt) | One row per in-scope ball: expectations, wicket cost, run value for batter, non-striker and bowler, WPA |
| `outputs/tables/phase3/`, `outputs/figures/phase3/` | Expectation calibration, wicket cost and its calibration, zero-mean checks, player value tables |
| `outputs/ball_value_adjusted.csv` (not committed; rebuilt) | Opposition-adjusted run value per ball, with the tiers and groups used |
| `outputs/tables/phase4/`, `outputs/figures/phase4/` | Opposition model (cross-validation, coefficients), same-player gaps, test cases, eligibility |
| `outputs/tables/phase5/`, `outputs/figures/phase5/` | Split-half reliability, weights, ratings, draft rankings, volume checks |
| `outputs/tables/phase6/`, `outputs/figures/phase6/` | Variant ranks, stability classes, guarded final order, absences, evidence card |
| `outputs/final_rankings.csv` | **The submission:** top 10 batters and bowlers with player_id, name, team(s), evidence, label |
| `outputs/final_rankings_top15.csv`, `final_rankings_eligible.csv` | Ranks 1-15, and every eligible player |
| `outputs/report/` | `SupeRR_Selector_Methodology_Report.pdf` (the submission), `methodology_report.md`, its figures (300 dpi) and `crosscheck.md` |

## Status

- Phase 1 (data audit): done — `outputs/data_audit.md`.
- Phase 2a (pressure definition proposal): approved — `outputs/phase2a_pressure_proposal.md`.
- Phase 2b (implementation and validation): approved 2026-11-17 — `outputs/phase2b_pressure_validation.md`. Single high-pressure cut locked.
- Phase 3a (performance measurement proposal): approved 2026-11-17 — `outputs/phase3_performance_proposal.md`.
- Phase 3b (run value: implementation and sanity checks): approved 2026-11-17, wicket-cost calibration (option B) added — `outputs/phase3_value_validation.md`.
- Phase 4a (fair comparison layer proposal): approved 2026-11-17 — `outputs/phase4_fairness_proposal.md`.
- Phase 4b (opposition adjustment and eligibility): approved 2026-11-17 — `outputs/phase4_fairness_validation.md`.
- Phase 5a (ratings proposal): approved 2026-11-17 — `outputs/phase5_rating_proposal.md`.
- Phase 5b (ratings and draft rankings): approved 2026-11-17 — `outputs/phase5_rating_validation.md`.
- Phase 6 (robustness, option B, tie-break, stability guard): approved 2026-11-17 — `outputs/phase6_robustness.md`.
- Phase 7 (final outputs and methodology report): done, awaiting review — `outputs/final_rankings.csv`, `outputs/report/methodology_report.md`.
- Phase 8 (final verification): `outputs/phase8_verification.md`.
- Phase 9 (submission PDF): `outputs/report/SupeRR_Selector_Methodology_Report.pdf`.
