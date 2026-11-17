# Phase 5a — Ratings and draft rankings: proposal

No code yet. This turns the opposition-adjusted run value from Phase 4 into one rating per eligible player and a draft top 10 per role. It follows the design you set (D31). Decisions needed are in §10.

## 1. The proposal in five lines

1. **Headline rating (your option A):** a player's pressure ability is the leverage-weighted average of their opposition-adjusted run value on high-pressure balls only, shrunk towards the population mean by empirical Bayes.
2. **Leverage enters once**, as the weight on each pressure ball. Cap at the 99th percentile or √LI: **chosen by which gives the more reproducible rating** (split-half test), not by taste.
3. **Three components** are measured for every player: pressure value, uplift over their own usual level, and consistency across matches. Each is shrunk once, and they are **combined with weights equal to their measured split-half reliability**. No fixed weights.
4. **Uncertainty is shown, not hidden:** match-bootstrap intervals, and for each player the probability of being in the top 10.
5. **Volume check:** evidence that the ranking is not just tracking how many pressure balls a player had.

## 2. What goes in

| Input | From |
|---|---|
| Opposition-adjusted run value per ball (`v_bat_adj`, `v_bowl_adj`) | Phase 4 |
| High-pressure flag (top 25% of balls by LI, one cut) and LI | Phase 2 |
| Eligible players (≥ 100 pressure balls in ≥ 10 matches): 137 batters, 139 bowlers | Phase 4 |

Units throughout: runs per 100 balls against a league-average player in the same match state, on the common IPL + T20I scale.

## 3. Component 1: pressure value (the headline)

> **Pressure value** = Σ (weight × adjusted value) ÷ Σ weight, over the player's high-pressure balls, × 100.

**The leverage weight.** LI on pressure balls has a median of 1.6, a 99th percentile of 7.8 and a maximum of 33.8.

| Weighting of pressure balls | Share of all weight on the top 1% of pressure balls | Effective sample size (share of balls) | Largest ÷ smallest weight |
|---|---|---|---|
| Raw LI | 6.2% | 65% | 29 |
| **LI capped at the 99th percentile of all balls (4.14)** | 2.3% | 86% | 3.6 |
| **√LI** | 2.5% | 94% | 5.4 |
| Equal (reference) | 1.0% | 100% | 1 |

- Raw LI would let a few last-over balls dominate and waste a third of the sample. Both candidates fix that.
- **How I propose to choose:** compute pressure value under each candidate and pick the one with the **higher split-half reliability** (§6). If they are within the test's own noise, choose √LI, which keeps more of the sample and has no arbitrary threshold. Equal weighting is reported as a reference.

**Shrinkage (empirical Bayes):**

> shrunk value = population mean + (observed − population mean) × τ² ÷ (τ² + s²)

- **s** is the player's own standard error from a **match bootstrap** (resample the player's matches with replacement, 1,000 times), so that balls within a match are not treated as independent.
- **τ** is the real spread across eligible players: observed variance minus the average s².
- **Population mean:** the leverage-weighted mean pressure value of eligible players (as you specified), not zero.
- A player with little or noisy evidence is pulled strongly towards the mean; a player with a lot of consistent evidence keeps most of their observed value.

## 4. Component 2: uplift (kept, reported, weighted by its reliability)

> **Uplift** = pressure value − the player's own usual value on ordinary balls, re-weighted to the bowling-phase mix of their pressure balls (D23).

- Both sides use opposition-adjusted values. Pressure balls carry the same leverage weight as in §3; ordinary balls are unweighted (their LI is low and similar).
- Shrunk the same way, towards zero.
- **What the data already suggest:** at 200 and 300 pressure balls the implied real spread of uplift is close to zero for both batters and bowlers (Phase 3 report, §11.5). If the split-half test agrees, uplift gets a weight near zero and the report says so plainly (§8).

## 5. Component 3: consistency across matches

The brief asks for consistency across matches. I propose a measure that is easy to explain and on the same footing for both roles:

> **Consistency** = the share of a player's matches in which they beat same-state expectation on pressure balls (leverage-weighted adjusted value above zero), counting only matches with at least 6 pressure balls.

- Eligible players have a median of 22 (batters) and 20 (bowlers) matches with pressure balls, and a median of 7 and 9 pressure balls per match. Matches with fewer than 6 pressure balls are too noisy to call "won" or "lost".
- Shrunk towards the population share, with a binomial standard error.
- **A known asymmetry:** batting value is skewed (many small positive balls, occasional large negative wickets), so the median batter "wins" 57% of such matches and the median bowler 47%. Ranks are computed within each role, so this does not cross roles.
- **Alternatives considered:** match-to-match standard deviation (rewards being reliably mediocre); average of the worst third of matches (harder to explain). Both can be added as Phase 6 checks.

## 6. Reliability and weights

**Split-half test** (your specification). For every eligible player, split their matches into odd and even (in date order), compute each component separately on both halves, and correlate the two halves across players. Spearman–Brown correction gives the reliability of the full-sample measure:

> reliability = 2r ÷ (1 + r), where r is the correlation between the halves

- Repeated over 200 random splits of matches (not just odd/even) to show how stable the estimate is.
- Run separately for batters and bowlers, and for each leverage weighting in §3.

**Combining the components:**

> rating = Σ reliability_k × (shrunk component_k ÷ τ_k), divided by Σ reliability_k

- Dividing by τ_k puts the three components on the same "real spread" scale, so a component's influence is set by its reliability, not by its units.
- **Shrinkage and reliability weighting do different jobs, so this is not double counting:** shrinkage says how sure we are about *this player*; the reliability weight says whether *this component* measures anything stable at all. A component with no real spread (τ near zero) contributes nothing either way.
- A component with negative or zero reliability gets weight zero.
- The published rating is converted back to runs per 100 balls (multiplied by the pressure-value τ), so the headline number stays interpretable.

## 7. Uncertainty and the volume check

**Uncertainty.** The match bootstrap is run on the whole pipeline from adjusted values to ranks:

- 90% interval for each player's rating;
- **probability of being in the top 10** (share of bootstrap replicates);
- the draft top 10 is shown with these, so a reader can see which places are secure and which are close calls.

**Volume check** (D16, D27). Shrinkage towards the mean lets players with many pressure balls reach the extremes. Phase 5 will report:

1. the rank correlation between rating and pressure-ball count among eligible players, next to the same correlation for unshrunk pressure value;
2. a **fixed-sample ranking**: every eligible player cut to the same number of pressure balls (random subsample, by match, repeated 200 times); how often each top-10 player stays in the top 10;
3. how many top-10 players come from the teams that chase most often (Phase 2b §5.4).

If the volume effect is strong, I will bring options back to you rather than adjust silently.

## 8. What the report will contain from Phase 5

- **Draft top 10 per role** with player_id, name, team(s), rating, interval, top-10 probability, pressure balls, and competition and opposition mix.
- **Overall top 10 v pressure top 10** (your request): who drops, who rises, and why. "Why" is decomposed into how the player's pressure balls differ from their other balls by phase, innings and opposition, so the reason is visible, not asserted.
- **An honest section on uplift ("clutch")**: what the brief asks for, how it was measured, its reliability for each role, and what follows for the ranking. If it does not reproduce, the conclusion is stated plainly: we cannot reliably tell which players raise their game under pressure; we can tell who performed well when it mattered, and the ranking is built on that.

## 9. Phase 6 checks registered so far

| Check | Varies |
|---|---|
| Hierarchical option B: pressure ability = overall ability + clutch deviation, the deviation shrunk by its estimated variance | Rating model |
| Logistic-model leverage; rule-based index; per-innings cut | Pressure definition |
| Batting-position-relative baseline; death-bowling role | Role |
| Opposition tier offsets instead of the ridge model; associates excluded | Opposition |
| Eligibility at 150 and 200 pressure balls (and 60) | Eligibility |
| ×1.25 on knockout matches | Match importance |
| Other leverage weighting (whichever of cap / √LI is not chosen), and equal weights | Weighting |
| A recency check (later seasons weighted more) | Time |

## 10. Decisions needed

| # | Question | My recommendation |
|---|---|---|
| 1 | Leverage weight on pressure balls | Chosen by higher split-half reliability between the 99th-percentile cap and √LI; √LI if tied |
| 2 | Shrinkage | Empirical Bayes per component, match-bootstrap standard errors, prior mean = population mean (zero for uplift) |
| 3 | Consistency measure | Share of matches (≥ 6 pressure balls) beating same-state expectation, shrunk |
| 4 | Combining components | Weights = split-half reliability (Spearman–Brown), components on a common "real spread" scale, negative reliability → zero weight |
| 5 | Uncertainty | Match-bootstrap 90% intervals and top-10 probabilities in the draft rankings |
| 6 | Volume check | The three checks in §7, with options brought back to you if the effect is strong |
