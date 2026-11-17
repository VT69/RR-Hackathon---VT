# Decision log

One entry per decision: what was decided, the alternatives, why, and the evidence. Status is **Proposed** until Vaibhav approves it, then **Approved** with the date.

## Phase 1 — Data audit

### D1. Data semantics for balls, runs and wickets — *Verified fact, implemented*
- **Rule.** Legal ball = not a wide and not a no-ball. Ball faced = not a wide. Bowler runs = bat runs + wide runs + no-ball penalty. Bowler wickets = Caught, Bowled, LBW, Stump, Hit Wicket. Team wicket = `is_wicket` (includes run-outs, retire-outs, obstruction). Retired hurt is not a wicket.
- **Key finding.** `extras_type = "other"` is a second no-ball label, not a miscellaneous extra.
- **Evidence.** 30,622 of 30,626 completed overs have exactly six legal balls under this rule (217 break if `other` is treated as legal). Ball-level totals reproduce 803 of 805 checkable official results; the two misses are source-data errors (a missing delivery in match 14130, a wrong result string in match 20190). See `outputs/data_audit.md` §4.
- **Citation rule (2026-11-16).** The reconciliation is the only validation cited in reports. Season figures recalled from memory are not cited anywhere.
- **Correction.** The first version of the check reported 804 of 807. It included two tied matches whose result text quotes the super-over margin; removing them gives 803 of 805.

### D2. Player key and name registry — *Approved 2026-11-16*
- **Decision.** Key everything on `player_id`. Names come from `competition_team_player_mapping.csv`, since the pack contains no players table.
- **Output rule.** Every ranking output carries `player_id` + name + team(s) so shared names can be told apart. Rashid Khan means id 4722.
- **Why.** Seven names are shared by two ids; ids are fully consistent across IPL and T20I.
- **Open point.** Vaibhav is checking the Unstop dropdown spellings.

### D3. Competition scope — *Approved 2026-11-16*
- **Decision.** IPL + T20I only. The two A-team events (30 matches, 6,985 balls) are excluded from ratings **and** from model training.
- **Alternatives.** (a) Include them as T20 data — rejected: the brief names IPL and T20I, and the standard is lower (7.8–8.0 runs/over). (b) Use them only to train models — marginal gain (3.5% of balls) for added explanation cost.

### D4. Super overs — *Approved 2026-11-16*
- **Decision.** Innings 3 and 4 are excluded from match state, models and ratings. To be stated in the report's limitations.
- **Alternatives.** Rate them as maximum-pressure balls — rejected: only 6 of 9 tied matches have the data, and 38 balls cannot support a model.

### D5. Rain-affected and incomplete matches — *Approved 2026-11-16*
- **Decision.**
  - Matches with a `revised_target` (23): use the revised target and revised overs for the chase.
  - Shortened with no flag (3: matches 15795, 31278, 48514): scheduled overs inferred from the first-innings length (12, 16 and 12 overs), documented in `outputs/data_audit.md` §5.
  - DLS result but no revised target (4) and the DLS tie (18532): keep the first innings, drop the second innings from pressure and win-probability work.
  - No-result / abandoned with balls (24): drop from win-probability training (no outcome); keep the balls for run/wicket expectation baselines.
  - First innings cut short by rain mid-way: the batting side did not know the true length, so those innings are dropped from win-probability training.
- **Standing rule.** Balls remaining and target in every downstream model use the revised or inferred overs and the revised target, never the nominal 20 overs.
- **Alternatives.** Drop all 38 rain matches — simpler, loses about 4% of matches including the IPL 2023 final.

### D6. Carried into later phases (flagged by the audit)
- Scoring inflation, Impact Player, knockout importance and associates in the pressure model → Phase 2a proposal (`outputs/phase2a_pressure_proposal.md` §5).
- Opposition-quality adjustment of player ratings → Phase 4.
- Whether a retire-out counts against the batter → Phase 3.

### D7. Environment — *Approved 2026-11-16*
- **Decision.** Project virtual environment in `.venv`, all versions pinned in `requirements.txt`, tests run under `pytest`.

## Phase 2 — Pressure definition

Proposal: `outputs/phase2a_pressure_proposal.md`. Evidence: `outputs/phase2b_pressure_validation.md`.

### D8. Core definition: leverage — *Approved 2026-11-17*
- **Decision.** Pressure on a ball = the expected absolute swing in win probability across that ball's possible outcomes, relative to an average ball (Leverage Index). A transparent rule-based index is kept as a companion for explanation and as the alternative definition in Phase 6.
- **Alternatives rejected.** Rule-based index alone (arbitrary weights, calls lost causes "high pressure", no first-innings form); personal-pressure measures (differ for batter and bowler on the same ball, hard to validate).

### D9. Win-probability candidates and selection — *Approved 2026-11-17*
- **Decision.** Four candidates on identical information: logistic regression, XGBoost and LightGBM (both with monotonic constraints), and an exact dynamic-programming model over the state grid (runs × balls × wickets) with state-dependent outcome probabilities including wickets and extras. No Monte Carlo.
- **Selection.** Held-out seasons only (train on four, test on the fifth): log-loss, Brier score, calibration, plus smoothness of the resulting leverage between neighbouring states.
- **Cross-fitting.** Every ball's win probability and leverage comes from models trained without its season.

### D10. Context in the win-probability model — *Approved 2026-11-17, scoring environment revised and confirmed the same day*
- **Scoring environment.** Runs per over for that competition type and season (plain season average). Justification: this is a descriptive task (how pressured was each past ball), not a forecast, so the season's scoring level is legitimate context. It contains no match results.
- **The leave-one-out leak (report appendix item).** The first approved version excluded the match itself. That makes the value different for almost every match (817 distinct values across 837 matches) and a mirror of the match's own total, so tree models used it to recognise individual matches. Held-out log-loss with it against the plain season average is in `outputs/tables/phase2b/ablations.csv` and the Phase 2b report, Appendix A. A coin flip scores 0.6931; the leaky tree models scored worse than that. Vaibhav confirmed the plain average and asked for the demonstration to stay in the final report's appendix.
- **Lesson applied from here on.** No model feature may take a value that is nearly unique to a match.
- **Associates.** A flag for T20Is involving an associate. Whether those balls count towards ratings is a Phase 4 decision.
- **Knockouts.** No multiplier in the headline definition. Knockout records are reported separately; ×1.25 is a Phase 6 sensitivity test.

### D11. High-pressure share and weighting — *Approved 2026-11-17*
- **Decision.** The top 25% of balls by Leverage Index are "high pressure". Leverage also weights balls inside the rating. Where the line is drawn: D16. Shape of the weight: D17.

### D12. Training set for win probability — *Approved 2026-11-17*
- **What changed from D5.** D5 planned to drop from training only the innings known to be interrupted mid-way. The delivery timestamps cannot reliably show when a revision became known, so **every rain-affected match (41) is left out of win-probability training**. They are still scored, using revised overs and target.
- **Also excluded from training.** Matches level after 20 overs (7 in scope) and matches without a winner. 774 of 837 in-scope matches train the models.
- **Target unknown, second innings not scored (6 matches).** 20182, 23674, 24822, 28334 (DLS result, no revised target), 18532 (DLS tie), and 48610, where the "revised target" is a par score computed after play stopped (the chasing side batted on past it).
- **Scheduled overs inferred (5 matches).** 15795, 31278, 48514 (no flag at all) and 16754, 23674 (rain-affected, first-innings length not recorded): the first innings ended on an over boundary with wickets in hand.

### D13. Win-probability model: dynamic programming with recalibration — *Approved 2026-11-17*
- **Decision.** DP + recalibration is the headline model. Leverage from the logistic model is a Phase 6 sensitivity check.
- **Evidence.** Held-out seasons, 185,070 balls from 774 matches: `outputs/tables/phase2b/wp_model_comparison.csv` and the Phase 2b report §3. Logistic and DP + recalibration are statistically tied overall; the two tree models are less accurate and about 20% of their leverage is noise between neighbouring states.
- **Why DP over logistic.** (1) It is more accurate in the last 12 balls of a chase, where leverage is highest, though that gap alone is not statistically significant. (2) It applies the laws exactly at the end of an innings. (3) Its leverage is smooth. (4) Every probability is an explicit sum over outcome likelihoods.
- **Recalibration.** The raw DP is over-confident because it treats balls as independent. A two-parameter logistic rescaling per innings, fitted on other seasons, roughly halves the calibration error and keeps the surface smooth.
- **Outcome model.** LightGBM multiclass over 13 outcome classes (runs, wicket, wide/no-ball combinations), 80 trees; reduced from 150 to keep the full run near 20 minutes.

### D14. First-innings states assume an equal-length chase — *Approved 2026-11-17*
- While the first innings is in progress a later rain reduction of the chase is unknown, so first-innings win probability assumes the chase will last as many balls as the first innings. The chase itself uses the revised overs and target (D5).

### D15. Context features kept and dropped — *Approved 2026-11-17*
- **Kept.** Scoring environment, IPL flag, associate flag.
- **Honest note on the scoring environment.** After the re-run without the Impact Player flag, removing the environment adjustment costs LightGBM only 0.0005 log-loss (the earlier run showed 0.0051). It is kept because IPL scoring rose from 8.54 to 9.88 runs per over and the DP needs a scoring level per season, not because it buys much accuracy. Raised with Vaibhav at the Phase 3a checkpoint.
- **Impact Player flag: dropped.** Tested; effect negligible. Evidence: the "+ Impact Player flag" rows of `outputs/tables/phase2b/ablations.csv` and `impact_wicket_cost.csv` (a wicket costs about 2% less win probability in the first innings and no less in a chase). The final report gets one line on this.
- **Team strength and venue: not included.** Negligible gain in a simple model; a match fingerprint in a flexible one.

### D16. High-pressure flag: one cut for the whole match — *Locked 2026-11-17*
- **Decision.** A ball is high pressure if it is in the top 25% of all scored balls by LI. This reverses Vaibhav's earlier lean towards a separate cut per innings.
- **Why.** (1) The role concern is resolved: batting positions 1–8 all have 24–26% of their balls flagged under one cut. (2) A separate first-innings cut would label balls with LI of about 0.95, below an average ball, as "high pressure", which is not defensible.
- **Framing.** "Pressure concentrates in chases" is reported as a finding, not hidden.
- **Before locking.** Vaibhav asked to see the share of pressure balls by bowling phase and how far each player's share depends on batting first v chasing. Evidence: `outputs/tables/phase2b/exposure_*.csv` and the Phase 2b report §5.
- **Phase 6.** The per-innings cut is a sensitivity check.
- **Carried to Phase 5 (Vaibhav, at lock).** Exposure correlates with team strength (India 17%, Kolkata Knight Riders 19% against Mumbai Indians 31%), so shrinkage must not end up rewarding sample volume. Phase 5 must show that the final ranking is not simply tracking pressure-ball counts.

### D17. LI weight shape — *Agreed in principle 2026-11-17, value in Phase 5*
- The largest LI is about 34 and the 99th percentile about 4, so raw LI weights would let a handful of last-over balls dominate. Phase 5 compares a hard cap (for example the 99th percentile) against a softer transform such as √LI.

### D18. Phase 6 alternative definitions — *Approved 2026-11-17*
- Logistic-model leverage (same definition, different model), the rule-based index (different definition), the per-innings cut (D16), and the ×1.25 knockout multiplier (D10).

## Phase 3 — Performance measurement

Proposal: `outputs/phase3_performance_proposal.md`. Evidence: `outputs/phase3_value_validation.md`.

### D19. Requirements set by Vaibhav — *2026-11-17*
- Every pressure-v-usual comparison is against expected performance in the same situation, never raw strike rate or economy.
- Runs and wickets combine into one value per ball, in the same units for batting and bowling.
- Leverage enters each metric once. A metric that already scales with leverage (WPA) is never LI-weighted.
- Two baselines: the league-average expectation in the same state, and the player's own context-adjusted non-pressure level.
- Bowling mirrors batting. League-level sanity checks are shown after implementation.

### D20. Run value per ball with the DP exchange rate — *Approved 2026-11-17*
- **Decision.** Batting `v = (runs − xR) − c × (wicket − xW)`; bowling `v = (xR − runs charged) + c × (wicket − xW)`. Run value is the primary metric. WPA is the companion and is never LI-weighted.
- **Wicket cost `c`.** Win probability lost to a wicket divided by the win probability one run is worth, both from the Phase 2 DP table one ball on from the state.
- **Fallback (Vaibhav).** Where the denominator is near zero (settled matches), use a smoothed `c` by balls left × wickets in hand. Implemented as the mean first-innings cost by overs left × wickets in hand, gaps interpolated: a first innings has no target, so its cost reflects only the batting resources a wicket removes.
- **Cap.** The configured quantile of the reliable ratios (`value.cost_cap_quantile`).
- **Condition.** Before `c` is used in any rating, Vaibhav reviews it in representative states (`outputs/tables/phase3/wicket_cost_examples.csv`). Until then the threshold, fallback and cap are **provisional**.
- **Alternatives.** Runs and wickets kept separate (no single ranking); a fixed run cost per wicket (ignores the state; kept as a Phase 6 check); WPA as the main unit (leverage-scaled).

### D21. Expectation models — *Approved 2026-11-17 with conditions*
- **Decision.** Five expectations (runs off the bat, striker dismissed, non-striker dismissed, runs charged to the bowler, bowler-credited wicket), cross-fitted by season.
- **Features.** Balls remaining, wickets in hand, powerplay flag, balls already faced by the striker, free-hit flag, scoring environment, IPL flag, associate flag; first innings adds score against environment pace; the chase adds runs required and required rate relative to the environment.
- **Conditions set by Vaibhav, and how they are met.**
  - Bowler expectations use the same feature set as batter expectations: one shared feature list in `src/models/expected.py`.
  - The chase state is in the features: `runs_required` and `rrr_rel`.
  - No feature can fingerprint a match: `outputs/tables/phase3/feature_fingerprint_audit.csv`.
  - Held-out calibration is shown for the expectation models: `expectation_calibration*.csv` and the calibration figure.
- **Left out on purpose.** Player identity, role, bowler type, venue, opposition (Phase 4) and leverage.

### D22. Crediting rules — *Approved 2026-11-17 with one change*
- Striker dismissals are charged to the striker.
- **Changed by Vaibhav:** a non-striker run-out is charged to the **dismissed batter** as a wicket cost, with no ball faced added (the proposal had charged nobody). To keep the league average at zero, every delivery also carries the small expected run-out risk for the batter at the non-striker's end (`xw_ns`).
- Retired-out is charged to nobody. Run-outs are not credited to bowlers. Byes and leg-byes are credited to nobody.

### D23. Pressure performance, usual level and uplift — *Approved 2026-11-17*
- Pressure value = mean `v` over high-pressure balls. Usual value = mean over the rest. Uplift = pressure value minus the usual value re-weighted to the phase mix of the player's pressure balls. Both are against expectation.
- Matching is on bowling phase; innings is added only if the cell counts allow (`pressure_split_reliability.csv`).
- **Flag for Phase 5.** Uplift is a difference of two noisy numbers and is noisier than either. Shrinkage must account for that.
- Reporting unit: runs per 100 balls for batting and bowling.

### D24. Scoring environment tested on the DP — *2026-11-17*
- Vaibhav asked for the test on the chosen model, with the rule "keep the feature unless it clearly hurts; the justification is era validity". Result: the "no scoring-environment adjustment" row for `dp_platt` in `outputs/tables/phase2b/ablations.csv`.
- **Result.** Held-out log-loss 0.5165 with the adjustment, 0.5222 without (+0.0057). It clearly helps the DP and stays.

### D25. Phase 3b choices — *Approved 2026-11-17*
- **Season rescaling of expectations: confirmed.** One level factor per season x competition type x expectation, because tree models cannot extrapolate to a season that scores faster than any in training. Consequence, to be documented as the era adjustment: **run value is zero-mean within each competition-season.** A player is measured against the league of the season and competition they played in. Evidence: `expectation_season_factors.csv`, `value_by_season_before_recalibration.csv`.
- **Wicket cost: approved with option B (D26).** Settled threshold (0.05 points of win probability per run), first-innings fallback table and the last-ball rule approved; the cap is re-derived after calibration.
- **Own baseline matched on bowling phase only: confirmed** (D23).
- **Expectation model settings.** 150 trees x 7 leaves for runs, 100 trees x 4 leaves for wickets, chosen from five options on held-out seasons.
- **Overall lists pass the sanity check**, with two flags carried to Phase 4: Muzarabani (#1), Nawaz and Mahedi Hasan in the bowling top 10 are likely flattered by opposition quality and are test cases for the opposition adjustment; several death bowlers in the bottom 10 (Jordan, Curran, Nortje, Alzarri Joseph, Shepherd) are re-checked after Phase 4.

### D26. Wicket-cost calibration (option B) — *Approved 2026-11-17, implemented*
- **Why.** An event study of runs actually lost after a wicket matched the DP cost on average but showed it about 30% too high in the first five overs and about 0.9 runs too low in the last two.
- **How.** Factor by overs left = runs actually lost / model cost, fitted on full-length, unaffected first innings. Made monotone (never rises with overs left) by isotonic regression and smooth by joining the centres of its blocks with straight lines. Cross-fitted: for season Y both the runs-lost baseline and the factors use the other seasons only. The cap (99th percentile) is re-derived after calibration.
- **Per innings: not possible on evidence.** In a chase the innings stops at the target, so the runs lost after a wicket are capped by the target and cannot be measured the same way. The first-innings factors are applied to both innings. Assumption stated in the report.
- **Vaibhav's premise, checked.** The request expected the fix to remove a systematic opener penalty. Rescaling the cost within a state cannot shift a group whose dismissal rate equals the expectation in that state; it changes only players who are out more or less often than expected. The batting-position gap (openers about −4 per 100 balls) is in the runs part, among set openers in the middle overs, not in wicket pricing. Evidence: `calibration_effect_by_batting_position.csv`, `calibration_effect_batters.csv`.

### D27. Phase 5 design requirements — *Set by Vaibhav 2026-11-17*
- Report the uplift reliability test (observed spread v standard error) for bowlers as well as batters.
- Weight rating components by their **measured reliability**: split-half (odd v even matches) correlation for pressure value, uplift and consistency, plus empirical-Bayes shrinkage of each. **No fixed weights** such as 50/30/20.
- Report framing: uplift is measured as the brief asks; if it does not reproduce for batters, say so and rank mainly on pressure value against same-state league expectation.
- Shrinkage must not end up rewarding sample volume; show the final ranking is not just tracking pressure-ball counts (from D16).

## Phase 4 — Fair comparisons

Proposal: `outputs/phase4_fairness_proposal.md`. Evidence: `outputs/phase4_fairness_validation.md`.

### D28. Wicket-cost calibration confirmed — *Approved 2026-11-17*
- Option B approved, including one set of factors for both innings. The chase assumption is stated in the report.
- The remaining opener gap is a scoring-rate effect, not wicket pricing. The batting-position-relative version stays a Phase 6 check.

### D29. Opposition and competition strength — *Approved 2026-11-17 with one change*
- **Decision.** Ridge regression of ball value on batter and bowler identity, fitted on all balls; the penalty is chosen by cross-validation on held-out matches.
- **Change (Vaibhav).** Add tier effects for IPL, T20I top-8, other full members and associates, lightly or not penalised, so each player shrinks towards their tier's mean instead of the global mean. Otherwise weakly linked associate batters are shrunk to "average" and the bowlers who face them stay flattered.
- **Acceptance test.** The same-player gaps (+17.0 against the four other full members; +9.5 against T20I top-8 v IPL) should mostly close. Show before/after, and where Muzarabani, Nawaz, Mahedi Hasan and the five death bowlers move.

### D30. Pooled ranking, role, eligibility, shrinkage — *Approved 2026-11-17*
- **One pooled IPL + T20I ranking** on a common scale. Limitation to state: each player gets a single rating across 2022–26, although players change over time.
- **Batting position:** no extra adjustment; position-relative version in Phase 6.
- **Eligibility:** at least 100 pressure balls across at least 10 matches; players below are listed as "insufficient evidence". 150 and 200 are tested in Phase 6.
- **Shrinkage once, in Phase 5,** with match-bootstrap standard errors and a volume check.
- **Associates** are included, adjusted (settles D10's open point).

### D31. Phase 5 design — *Set by Vaibhav 2026-11-17*
- **Headline (A):** pressure ability estimated from pressure balls only (opposition-adjusted, leverage-weighted; cap v √LI pending, D17), with empirical-Bayes shrinkage towards the population mean. A descriptive ranking of who delivered when it mattered.
- **Phase 6 check (B):** hierarchical version, pressure ability = overall ability + clutch deviation, the deviation shrunk by its estimated (near-zero) variance.
- Components weighted by split-half reliability. Uplift is kept and reported, weighted by its reliability (likely near zero).
- The report needs a table of overall top 10 v pressure top 10 (who drops, who rises, and why) and an honest section on the uplift / clutch finding.

### D32. Phase 4b approvals — *Approved 2026-11-17*
- **Tier split approved:** each T20I tier is split into IPL-linked and T20I-only players (`opposition.split_groups_by_ipl`), justified by better held-out prediction (batting 7.6299 v 7.6318 without it; bowling 7.5791 v 7.5810).
- **Follow-up fairness check:** ratings with v without the split for the top T20I-only players (Babar Azam, Mohammad Rizwan, Shaheen Shah Afridi, Haris Rauf and the highest-volume T20I-only players generally), to confirm the split is not pushing strong high-volume players down through their group mean.
- **Report note:** the "other four full members minus top-8" batter gap flips from +1.2 to −3.3 after adjustment (within one standard error), a possible slight overcorrection.
- **Before Phase 7:** a fresh `python -m src.pipeline --force` run end to end.

## Phase 5 — Ratings

Proposal: `outputs/phase5_rating_proposal.md`. Evidence: `outputs/phase5_rating_validation.md`.

### D33. Rating design — *Approved 2026-11-17 with changes*
- **Leverage weight within pressure balls:** compare three options by split-half reliability of pressure value: uniform weight, LI capped at the 99th percentile (4.14), and √LI. Prefer the simplest if they tie (settles D17).
- **Consistency:** share of matches (≥ 6 pressure balls) in which the player beats same-state expectation, **relative to the league base rate** (not 50%), shrunk with a **beta-binomial prior**. The base rate is reported.
- **Composite:** components **z-standardised before reliability weighting**. Report the correlation between pressure value and consistency, and the final effective weights as a formula.
- **Ordering:** rank by the shrunk point estimate. 90% intervals and P(top 10) from a match-level bootstrap that **re-estimates shrinkage inside each resample**; use them to justify close adjacent pairs.
- **Volume checks** as proposed, including equal-ball truncation.
- **Checkpoint contents:** draft top 10 per role (player_id, name, team(s), pressure balls, matches, pressure value raw and shrunk, 90% interval, consistency, P(top 10), overall-value rank v pressure rank); overall top 10 v pressure top 10 (who rises, who drops, why); ranks 11–15 and how close; split-half reliabilities of every component.

### D34. Phase 5b implementation choices — *Proposed 2026-11-17, need Vaibhav's confirmation*
- **Leverage weight: uniform** within pressure balls. Split-half reliability of pressure value (both roles averaged): uniform 0.262, √LI 0.253, cap 0.231, all within one SD (about 0.09), so the simplest wins (D33 rule).
- **Pooled standard errors:** each player's standard error = median match-clustered noise level / √(pressure balls). Individual standard errors from 20–50 matches are too noisy and let lucky-low errors escape shrinkage.
- **Bootstrap correction:** inside each resample, real spread = observed variance minus twice the mean squared standard error (and likewise for the beta-binomial), because resampling adds a second dose of noise. Without it, some intervals missed their own point estimate.
- **Reliabilities:** pressure value 0.33 (batters) / 0.19 (bowlers); uplift 0.00 / −0.07 (weight zero); consistency 0.08 / 0.15. Consistency base rates 55.2% / 44.5%.
- **Composite:** batters 0.81 z(pressure) + 0.19 z(consistency); bowlers 0.57 / 0.43.

### D35. Phase 5b approvals and Phase 6 brief — *Set by Vaibhav 2026-11-17*
- **Pooled match-clustered standard errors: confirmed.**
- **Intervals:** explain exactly what the bootstrap correction does; if it is ad hoc, switch to **posterior simulation** (draw each player from their empirical-Bayes posterior, normal approximation) for 90% intervals and P(top 10).
- **Reliabilities** are reported both as the raw split-half correlation and Spearman–Brown corrected (2r / (1 + r)).
- **Hierarchical option B** (pressure ability = overall ability + deviation, deviation shrunk by its estimated variance) is run.
- **Tie-break rule:** A stays the headline. Where adjacent players in ranks 1–15 of A are statistically indistinguishable (overlapping 90% intervals **and** P(top 10) within about 0.15), they are ordered by B. Show which positions change and why. Report the A-v-B top-10 overlap for both roles.
- **Notable absences table** (both roles): Bumrah, Rashid Khan, Russell, Kohli, Patidar, plus any overall top-5 player missing the pressure top 10: pressure rank, raw and shrunk value, interval, P(top 10), pressure balls, and a plain-language reason from their pressure-ball record. Evidence only, no reputation overrides.
- **Sensitivity analysis:** logistic leverage, rule index, per-innings cut; knockout ×1.25; eligibility 150 / 200; IPL-only / T20I-only / pooled; leave-one-season-out; batting-position-relative and death-bowler role checks. Rank-stability heatmap; each top-15 player classified stable / moderately stable / fragile. End with a recommended final ordering under the tie-break rule.
- **Sooryavanshi evidence card** shown prominently.
- Fresh `--force` run still due before Phase 7.

## Phase 6 — Robustness

Report: `outputs/phase6_robustness.md`.

### D36. Phase 6 implementation choices — *Proposed 2026-11-17, need Vaibhav's confirmation*
- **Posterior simulation** replaces the bootstrap correction (which was a method-of-moments fix assuming resampling noise equals data noise). 4,000 draws per player from the empirical-Bayes posteriors (normal; beta for consistency); hyperparameters held at their estimates.
- **Reliability correction:** the Phase 5 figures (0.33 batters, 0.19 bowlers) were already Spearman–Brown corrected; raw split-half r = 0.20 and 0.11. Both are now reported.
- **Option B** = shrunk phase-matched usual level + shrunk uplift. Top-10 overlap with A: 5/10 batters, 4/10 bowlers.
- **Tie-break applied pairwise:** all 14 adjacent pairs in ranks 1–15 are indistinguishable in both roles, so a chained reading would make B decide the whole order. Players keep A's order when directly distinguishable; otherwise B orders them (greedy order respecting every distinguishable pair).
- **Open:** the tie-break promotes fragile players (Ashutosh Sharma; Ellis, Wood) over moderately stable ones (Nissanka; Southee, Hosein). Keep, or add a stability guard.

### D37. Phase 6 approvals and the stability guard — *Approved 2026-11-17*
- **Pairwise tie-break: approved.** Posterior simulation and raw + corrected reliabilities: approved.
- **Stability guard (Vaibhav):** the tie-break may never promote a fragile player above a stable or moderately stable one, and only stable or moderately stable players may be in the final top 10.
- **Stability computed over the 16 perturbation variants only:** logistic leverage, per-innings cut, √LI and cap weights, knockouts ×1.25, eligibility 150 and 200, IPL only, T20I only, associates excluded, the five leave-one-season-out runs, role-relative. The rule index (rejected in Phase 2b) and option B (a different question) are shown as "alternative definitions", not counted.
- **Report framing for Phase 7:** players high in A but low in B (Cummins 88th, Southee 104th, Hosein 124th in B) "delivered under pressure in this window; we don't claim it predicts future pressure performance", consistent with the uplift finding. Insight: Narine is a bottom-10 batter overall but the #1 pressure bowler.
- **Heatmap:** no empty band under the subtitle; columns grouped (definition / weighting / eligibility / competition / seasons / role / alternatives) with separators and labels; a "% of variants in top 10" column on the right.
- Then a fresh `--force` run, numbers reproduced, tests passing, and Phase 7.

## Phase 7 — Final outputs and report

Report: `outputs/report/methodology_report.md`. Cross-check: `outputs/report/crosscheck.md`.

### D38. Profile label instead of a per-player disclaimer — *Set by Vaibhav 2026-11-17, implemented*
- **Decision.** Every ranked player carries a descriptive `label`: **Pressure-specific** if their option B rank is worse than `report.framing_min_b_rank` (30): they delivered under pressure well beyond their usual level, which is what the brief asks for. **All-round** otherwise: A and B agree, elite under pressure and overall.
- **The caveat is stated once**, in the report's reliability section (§5.3): pressure-v-usual gaps do not reproduce across matches, so the labels describe 2022–26 rather than predict the future. It replaces the per-player "window" note of D37.
- **Short-record flag kept** as a separate `note` column (fewer than `report.short_record_pressure_balls` = 200 pressure balls): Sooryavanshi and Inglis.
- **Result.** Batters: Nissanka is Pressure-specific, the other nine All-round. Bowlers: Cummins, Axar Patel, Ferguson, Southee and Hosein are Pressure-specific.

### D39. Report production and number cross-check — *Set by Vaibhav 2026-11-17, implemented*
- **Figures** are drawn by the pipeline's report stage from the output CSVs only (`src/viz/report_plots.py`), one style, A4 width, 300 dpi, with a one-line takeaway caption in the report.
- **Ranking tables in the report** are rendered from `final_rankings_top15.csv` by the same function the cross-check uses (`check_report.ranking_rows`), not typed.
- **Cross-check.** `src/report/check_report.py` + `report_claims.py`: each checked number is a phrase from the report with its value recomputed from the tables or config; result is ok / MISMATCH / MISSING, and every number no check covers is listed. Run standalone or at the end of the pipeline.
- **Corrections made while writing the report** (found by checking every number against the current tables): the "eligibility 200 removes three batters below the bar" claim from Phase 6 was wrong (two fall below the bar; Klaasen slips to 11th); excluding associates keeps 8, not 9–10, of the bowling top 10; the uplift sample range is 100–888 pressure balls, not 100–700; India's batting pressure share is 17.1% and KKR's 18.9% on the current run.
- **Length.** Main body renders to 14 A4 pages at 10 pt (appendices 5 more): the executive summary was cut to half a page, and the sensitivity table and three supporting figures moved to the appendices.

## Phase 9 — Submission PDF and public repository

### D40. Report fixes, PDF build and repository layout — *Set by Vaibhav 2026-11-17, implemented*
- **Ranking tables show what sets the order.** Each table carries Rating (the composite) and Consistency next to shrunk value, and ‡ marks positions set by the tie-break, with a generated footnote naming the players that could not be told apart and their option A and B ranks. `final_rankings*.csv` gain `rating` and `set_by_tie_break`.
- **One stated pool per rank.** Overall ranks in §4.5 are among the eligible players (137 batters, 139 bowlers), the same pool as the pressure ranking. The §3.2 test cases and the Narine insight state their own pool (players with at least 500 balls).
- **Sooryavanshi's raw value is decomposed** (runs, wicket term, non-striker run-out term, opposition adjustment), computed in `pressure_record` and checked to sum to the raw value. His wicket term is positive despite 8 dismissals against 7.6 expected, because the dismissals came where wickets were cheap.
- **Author block** under the title; repository link placeholder `[REPO_LINK]` until the URL is known.
- **PDF** built by `python -m src.report.build_pdf` (also the pipeline's last step): Markdown → styled HTML (title page, clickable contents, figures embedded at 200 dpi) → PDF printed by headless Edge/Chrome, with page numbers and running header from CSS page-margin boxes and bookmarks from the browser's document outline. Settings in `config.yaml` → `report.pdf`.
- **Repository.** Raw data moves to `data/raw/` (ignored by git); `.gitignore` also excludes `.venv`, caches and the three ball-level CSVs (40–60 MB each). `pypdf` and `pypdfium2` are pinned for the PDF checks.
