# Phase 2a — Pressure definition: proposal

No code has been written for this phase. This is the proposal to approve, amend or reject. Decisions needed are P1–P7, summarised in §8.

## 1. Recommendation in one sentence

> **Pressure on a ball is how much the match result is expected to swing on that ball.**

I recommend your leverage hypothesis as the core, with three changes: a proper first-innings win-probability model instead of a par-score add-on, no knockout multiplier in the headline definition, and a transparent rule-based index built alongside it as an explanation and as an independent cross-check.

Three terms, used the same way throughout:

- **Win probability (WP):** the chance the batting side wins from the current match state, as estimated from how matches in similar states ended.
- **Leverage:** the average size of the WP change this ball can cause, weighting each possible outcome (dot, 1, 2, 3, 4, 6, wicket, wide, no-ball) by how likely it is in that state.
- **Leverage Index (LI):** leverage divided by the leverage of an average ball. LI = 2 means the ball matters twice as much as a typical one.

## 2. Why leverage, and not a rule

A rule such as "required rate above 10 with 5 wickets down" gets one thing badly wrong: **pressure is not highest when the task is hardest; it is highest when the result is most in doubt.**

| Situation | Rule-based view | Leverage view |
|---|---|---|
| Need 60 off 12 balls, 8 down | Maximum pressure | Near zero — the match is already lost |
| Need 9 off 6 balls, 5 down | High | Very high — every ball moves the result |
| Need 20 off 30 balls, 9 down | Low (easy rate) | High — one ball can end it |
| 25 for 4 after 5 overs, batting first | Not covered (no target) | High — another wicket is very costly |
| 190 for 2 after 17 overs, batting first | Not covered | Low — the innings is already safe |

The leverage column is what the definition should produce; Phase 2b reports the measured LI for each of these five situations.

Leverage also has two properties the brief asks for directly:

- **It works for both disciplines.** Leverage belongs to the ball, so the batter and the bowler on that ball face the same pressure.
- **It depends only on the situation before the ball**, never on what the player then did. The pressure label cannot be contaminated by the performance it is used to judge.

## 3. How it is built

1. **Match state before each ball.** Innings, balls remaining, wickets in hand, score (first innings) or runs required (chase), plus the context features in §5. Balls remaining and target use the revised values in rain matches (decision D5).
2. **Outcome likelihoods.** A model of how likely each outcome is in that state for an *average* player: dot, 1, 2, 3, 4, 6, wicket, wide, no-ball. League-average on purpose: pressure describes the situation, not the people in it.
3. **Win probability.** Two models, because the inputs differ:
   - *Chase model:* WP from runs required, balls remaining, wickets in hand.
   - *First-innings model:* WP for the side batting first from current score, balls remaining, wickets in hand.
   - Both are constrained to behave sensibly: WP can only rise with wickets in hand, fall as runs required rise, and rise with score.
4. **Leverage.** For each ball, apply every possible outcome to the state, look up the new WP, and average the absolute WP changes using the likelihoods from step 2.
5. **Leverage Index.** Divide by the mean leverage over all in-scope balls.

## 4. The design choices

### P1. Core definition — leverage *(recommended)*
Alternatives considered:

- **Rule-based index only** (required rate v current rate, wickets, balls left, closeness). Easy to explain, but the weights are arbitrary, it fails on lost causes (table above), and it has no natural first-innings form.
- **Personal pressure** (new batter after a collapse, dot-ball build-up, bowling to a set batter). Real, but it differs for batter and bowler on the same ball and is hard to validate. A collapse is already inside leverage, because lost wickets move WP.
- **Hybrid** — leverage as the definition, with the rule-based index built as well. This is what I recommend. The rule index does two jobs: it explains what drives leverage in terms a judge recognises, and in Phase 6 it is the "different pressure definition" for the sensitivity test.

### P2. High-pressure cut or continuous weight — use both, for different jobs
- **A cut defines "high pressure":** the top 25% of balls by LI. The brief asks for performance in high-pressure situations *compared with usual levels*, which needs two groups of balls. It is also one sentence a judge can repeat.
- **A continuous weight is used inside the rating:** within those balls, each counts in proportion to its LI, so a ball at LI 4 counts more than one at LI 1.6. This avoids treating everything above an arbitrary line as equal.
- 25% is a starting value. I will show the LI distribution in Phase 2b before fixing it, and Phase 6 tests 15%, 20%, 25% and 30%.

### P3. First innings — its own WP model, on the same scale
- Your option of a WP model based on par is the right one, done directly: the model learns the chance of winning from (score, wickets, balls left) in the first innings. Par score and collapse are then implied rather than bolted on. 25 for 4 has low WP and high leverage without any special rule.
- **Known consequence:** first-innings balls will have lower LI on average, because the result is further away. The high-pressure set will be mostly chase balls. That is accurate, but it tilts exposure towards chasers.
- **Decision deferred to evidence:** in Phase 2b I will compute both the common-scale LI and an innings-normalised version (top 25% *within each innings*), show the split, and ask you to choose.

### P4. Which WP model — decided by held-out evidence
Candidates:

1. Logistic regression (baseline).
2. XGBoost with monotonic constraints.
3. LightGBM with monotonic constraints.
4. **Added by me: a ball-by-ball simulation model.** It uses the outcome likelihoods from step 2 and works backwards from the last ball to compute the exact WP for every (balls, runs, wickets) state. It needs no separate fitting, and its WP is guaranteed consistent with the outcome model.

Why add a fourth: leverage is a *difference* between WP at neighbouring states. Tree models produce WP in small steps, so differences between neighbours can be zero or jump, which makes leverage noisy even when the WP itself is accurate. The simulation model is smooth by construction. It may lose on raw accuracy because it ignores who is batting, which is why it is a candidate, not the choice.

Selection criteria, all on **held-out seasons** (train on four seasons, test on the fifth, rotating):

- Log-loss and Brier score.
- Calibration (reliability plot), overall and by innings and phase; isotonic or Platt correction if needed.
- Smoothness of the resulting leverage.

### P5. No leakage — cross-fitted WP
The WP used for a match comes from a model that never saw that season. Otherwise a flexible model can partly memorise results and understate uncertainty. Five models instead of one, same single command.

### P6. Match importance — no multiplier in the headline
See §5.3.

### P7. Tied matches and labels
The nine matches level after 20 overs are excluded from WP training (the label would be the super-over result, which D4 excludes). Their balls still receive a WP and LI from the model.

## 5. Your four carry-over questions

### 5.1 Scoring inflation — a 180 chase in 2022 is not a 180 chase in 2026
- **Proposal:** a *scoring-environment* feature — the average runs per over for that competition type in that season, computed leaving out the match itself. Required rate and current rate then enter the model both raw and relative to it.
- **Effect:** needing 10 an over is 1.17× the going rate in IPL 2022 (8.54) and 1.01× in IPL 2026 (9.88). The model sees them as different situations.
- **Trade-off for you to decide.** This uses the whole season's scoring level, including matches played later. It leaks no match results, and it is the standard way to adjust for era. The stricter alternative is a past-only trailing average, which is beyond criticism but lags badly when scoring jumps (IPL 2023 → 2024 rose 0.57 runs per over) and has no history for early 2022. I recommend the season average and will run the trailing version as a comparison.
- Venue is not included at first: 98 venues with thin samples, and in a chase the target already carries the pitch on the day. I will test a venue factor and keep it only if held-out accuracy improves.

### 5.2 Impact Player — a wicket is worth less in IPL 2023+
- **Proposal:** a flag for matches played under the rule (IPL 2023 onwards), which lets the model learn a different value of wickets in hand in that regime.
- **Check in Phase 2b:** at matched states, the WP cost of losing a wicket should be smaller under the rule. If the model shows no difference, the flag is dropped and I say so.
- **Not proposed:** tracking the exact moment each substitution happened. The data marks who was substituted but not when.

### 5.3 Knockouts — 25 in-scope matches, no World Cups
- **Recommendation: no multiplier in the headline definition.** Leverage already measures how much a ball matters to its match. How much the *match* matters is a separate question that 25 matches (2.9% of the in-scope 853) cannot answer or validate, and any number chosen would be arbitrary. A multiplier would also reward players for their team reaching the playoffs.
- **Instead:** (a) test in Phase 2b whether knockout balls behave differently from league balls at the same state; (b) report each top-10 player's knockout record as supporting evidence; (c) run a modest ×1.25 multiplier in Phase 6 to show whether it changes the rankings.
- If you prefer a multiplier in the headline, ×1.25 is the largest I could defend.

### 5.4 Associates — lower-scoring, less balanced games
- **In the pressure model:** a flag for T20Is involving an associate, so those games are judged against their own scoring level (7.7–7.9 runs per over against 8.6).
- **Known weakness:** the WP model is team-neutral, so it treats India v Hong Kong as an even contest at the start. Leverage is therefore somewhat overstated in mismatches. I will test adding a pre-match team-strength rating (built from earlier results only) and report whether it improves held-out accuracy.
- **Size of the issue:** 54 matches, 11,950 balls, about 6% of in-scope balls. Whether those balls count towards player ratings is the Phase 4 decision.

## 6. Weaknesses, stated up front

- **Role concentration.** High-leverage balls cluster in the last overs of close chases, so finishers and death bowlers get the most exposure. Phase 3 (judging each ball against what is expected in that situation) and Phase 4 (eligibility, shrinkage) have to deal with this.
- **Model dependence.** Pressure is only as good as the WP model. Mitigations: held-out validation, cross-fitting, and the rule-based index as an independent check.
- **Team-neutral and player-neutral.** It ignores who is batting or bowling and how strong the sides are.
- **First innings is inherently less certain**, so pressure there is lower and less precisely measured.
- **It measures stakes, not feelings.** A debutant's nerves in a dead game are invisible to it.

## 7. How I will show it separates pressure from ordinary play (Phase 2b)

1. **Distribution** of LI, overall and by innings, phase and competition.
2. **Outcome comparison**, high-pressure v ordinary balls: scoring rate, dismissal rate, boundary %, dot %, outcome variance. Shown overall *and within the same phase*, so the difference cannot be dismissed as "death overs are just different".
3. **Face validity:** a heat-map of LI across balls remaining × runs required, by wickets in hand; and the five situations in §2 with their actual LI.
4. **Match walk-throughs:** ball-by-ball WP and LI for 3–4 close finishes. Chosen from the data by rule (smallest margins, including at least one knockout, one T20I, one successful chase and one successful defence), not by reputation.
5. **Concentration:** the share of all WP movement that happens on the top 25% of balls.
6. **Against the rule-based index:** how often the two agree, and what kind of ball they disagree on.
7. **WP model evidence:** the comparison table and reliability plots from P4.

## 8. Decisions needed

| # | Question | My recommendation |
|---|---|---|
| P1 | Core definition | Leverage, with a rule-based index as explanation and cross-check |
| P2 | Cut or continuous | Top 25% defines "high pressure"; LI weights balls inside the rating; cut finalised after seeing the distribution |
| P3 | First innings | Own WP model; common scale v innings-normalised decided on Phase 2b evidence |
| P4 | WP candidates | Logistic, XGBoost, LightGBM, plus the simulation model; chosen on held-out seasons |
| P5 | Cross-fitting by season | Yes |
| P6 | Knockout multiplier | None in the headline; ×1.25 as a sensitivity test |
| P7 | Tied matches | Excluded from WP training, still scored |
| 5.1 | Scoring environment | Season average excluding the match; trailing past-only version as a comparison |
| 5.2 | Impact Player | Rule flag in the WP model, kept only if it shows an effect |
| 5.4 | Associates | Flag in the WP model; team-strength feature tested; rating inclusion left to Phase 4 |
