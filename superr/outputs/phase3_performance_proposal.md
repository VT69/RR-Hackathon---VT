# Phase 3a — Measuring performance: proposal

No code yet. This sets out how each ball will be valued, what it is compared against, and exactly where leverage enters. Decisions needed are in §11.

## 1. The proposal in five lines

1. Every ball gets **one number per player: its run value**. This is how many runs better (or worse) the player did than a league-average player would be expected to do **in that exact match situation**.
2. Runs are compared with expected runs, and wickets with the expected chance of a wicket, both from models of the situation.
3. A wicket is converted into runs at the **exchange rate the Phase 2 DP gives for that state**, so batting and bowling are both measured in runs.
4. **Leverage enters once**: it decides which balls count as pressure and how much each weighs. It is never inside the run value.
5. Each player gets two numbers: **pressure performance** (run value on high-pressure balls) and **uplift** (pressure performance minus the player's own level on ordinary balls, matched for context).

## 2. Why raw strike rate and economy cannot be used

High-pressure balls are different for everyone, not just for good players. Under the common-scale cut now adopted:

| | Ordinary balls | High-pressure balls |
|---|---|---|
| Batter strike rate | 138.3 | 145.1 |
| Striker dismissed, % of balls | 5.28 | 6.01 |
| Runs charged to the bowler per ball | 1.380 | 1.445 |
| Bowler-credited wicket, % of balls | 4.96 | 5.52 |
| Team runs per over | 8.78 | 9.25 |

(The 9.7 against 8.6 runs per over quoted earlier was for the per-innings cut. Under the single cut the gap is 9.25 against 8.78: smaller, but present in every phase.)

Compared on raw numbers, almost every batter would "score faster under pressure" and almost every bowler would "leak runs under pressure". The run value removes this, because each ball is judged against what is normal in its own situation.

## 3. Who is credited with what

| | Batter | Bowler |
|---|---|---|
| Runs | Runs off the bat | Runs off the bat + wides + no-ball penalty |
| Balls | Every delivery as striker (the runs part applies to balls faced, so not wides) | Every delivery bowled, including wides and no-balls |
| Wicket | The striker is out, by any mode except retiring | Caught, Bowled, LBW, Stumped, Hit Wicket |
| Not counted | Byes and leg-byes | Byes, leg-byes, run-outs |

Evidence and proposed rules for the edge cases:

- **Striker dismissals are 10,191 of 10,452 wickets (97.5%).** These are charged to the batter.
- **Non-striker run-outs: 258.** I propose charging these to nobody. The data cannot say whose call it was, and charging the striker or the non-striker would be a guess.
- **Retired out: 6.** A team tactic, so not charged to the batter. Retired hurt (11) is not a wicket at all.
- **Stumpings off wides: 25.** Not a ball faced, but the batter is out and the bowler earned it, so both are counted.
- **Byes and leg-byes: 4,533 runs (1.6% of all runs).** Credited to neither player.

Because of these rules, a batter's and a bowler's value on the same ball are not exact opposites. Each is charged only for what they control. The two formulas mirror each other (§5).

## 4. Expected performance in the same situation

Four expectations are needed for every ball:

| Symbol | Meaning |
|---|---|
| xR_bat | Expected runs off the bat |
| xW_bat | Chance the striker is dismissed |
| xR_bowl | Expected runs charged to the bowler |
| xW_bowl | Chance of a bowler-credited wicket |

**Model.** LightGBM, one model per expectation and per innings (regression for runs, classification for wickets). Cross-fitted by season exactly as in Phase 2: a ball in season Y is judged by models that never saw season Y.

**Features (the situation, and nothing about who is playing):**

| Feature | Why |
|---|---|
| Balls remaining (revised or inferred overs) | Teams attack more as the innings runs out |
| Wickets in hand | Fewer wickets means more caution |
| Runs required and required rate relative to the scoring environment (chase only) | The chase task sets the tempo |
| Score relative to environment pace (first innings only) | Sides ahead of or behind the rate behave differently |
| Powerplay flag, scaled for shortened matches | Fielding restrictions |
| Scoring environment (season × competition), IPL flag, associate flag | Same context as Phase 2 |
| **Balls the striker has already faced** | A new batter scores slower: strike rate is 101 on balls 1–3 and 154 on balls 12–21, and the gap holds inside each phase |
| **Free-hit flag** (previous delivery was a no-ball) | On 752 free hits the strike rate is 233 and the dismissal rate 0.27%, against 5.62% otherwise |

**Left out on purpose:**

- **Player identity, batting position, bowler type.** These describe the player or role, not the situation. Role-fair comparison is Phase 4.
- **Venue and opposition quality.** These belong to the Phase 4 fairness layer. In Phase 2b they did not help, and match-specific features are how the leak happened.
- **Leverage itself.** It is a function of the same state, so the models can already learn any pressure effect. The check in §9 confirms it.

**Leak rule carried from Phase 2b.** No feature may take a value that is nearly unique to a match.

**Validation on held-out seasons.** Predicted against actual, by tenth of the prediction range, by innings × phase, by season and competition, and by pressure flag. Each expectation is also compared with a simple season × phase average, which is the traditional baseline.

## 5. One value per ball

**Batting:** `v = (runs − xR_bat) − c × (wicket − xW_bat)`

**Bowling:** `v = (xR_bowl − runs charged) + c × (wicket − xW_bowl)`

Here `wicket` is 1 or 0 and `c` is the cost of a wicket in runs, in that state. Positive is always good for the player. The league average is zero in every situation by construction.

### The wicket exchange rate `c`, from the DP

The DP table already holds the win probability of every state. For the state a ball is bowled in:

> **c = (win probability lost if a wicket falls) ÷ (win probability one run is worth)**

Both parts are read from the table at the same number of balls left: the first by removing a wicket, the second by moving the score (or the runs required) by one run.

- **It varies with the state.** A wicket with two balls left costs almost nothing. A wicket with twelve overs left costs many runs. In a dead chase a wicket is nearly free; with the last pair at the crease it is very expensive.
- **It contains no leverage.** It is a ratio of two win-probability differences, so if a state mattered twice as much, both parts would double and `c` would not move.
- **Edge cases.** Where both parts are too small to divide reliably (the match is settled), `c` falls back to the expected runs the lost wicket would have produced over the rest of the innings, from the same outcome model. A cap stops rare extreme values. I will set the cap from the distribution of `c` and show it to you before using it.

### Alternatives considered

| Option | Verdict |
|---|---|
| Keep runs and wickets as two numbers | Cannot produce one ranking |
| A fixed run cost per wicket | Ignores that a wicket's cost depends on balls and wickets left. Kept as a Phase 6 check |
| Win probability added (WPA) as the main unit | Already scales with leverage, so a "pressure v usual" comparison of WPA mostly compares leverage, and the usual level is set by a handful of balls. Kept as a companion (§8) |
| **Run value with the DP exchange rate** | Recommended |

The recommended run value is WPA with the leverage divided out and expressed in runs: on any ball, `WPA ≈ (win probability one run is worth) × v`.

## 6. Where leverage enters

| Quantity | Unit | Where leverage enters |
|---|---|---|
| Expected runs and wickets | runs, probability | Nowhere |
| Wicket exchange rate `c` | runs | Nowhere (leverage cancels in the ratio) |
| Run value `v` | runs per ball | Nowhere |
| Usual level | runs per 100 balls | Only to exclude high-pressure balls |
| **Pressure performance** | runs per 100 balls | **Here, once:** the top-25% cut selects the balls and a function of LI weights them |
| Uplift | runs per 100 balls | Inherited from the two lines above |
| WPA (companion) | win probability | Inside the win-probability surface. Never weighted by LI |

The shape of the LI weight (hard cap at the 99th percentile against √LI) is the Phase 5 comparison you asked for. Phase 3 will store `v` and LI for every ball so that either can be applied.

## 7. Two baselines, two numbers per player

**Baseline A: the league in the same situation.** This is built into `v`.

- **Pressure performance** `P` = weighted average of `v` over the player's high-pressure balls. It reads as "runs per 100 balls better than an average player facing the same situations".
- **Usual level** `U` = plain average of `v` over the player's ordinary balls, on the same scale.

**Baseline B: the player's own ordinary level.**

- **Uplift** = `P − U*`, where `U*` is the usual level re-weighted to the same mix of contexts as the player's pressure balls.

**Why the re-weighting is needed.** Pressure balls and ordinary balls come from different parts of a match. 89% of pressure balls are in chases, and two-thirds of ordinary balls are in the first innings. A player can be better in one phase than another for reasons of skill. Without matching, a new-ball bowler who also bowls at the death in tight chases would show a "pressure drop" that is really a phase effect.

**How.** Split each player's balls into cells by bowling phase (powerplay, middle, death). Compute the usual level in each cell, then average the cells using the shares of the player's pressure balls: `U* = Σ share_of_pressure_balls(cell) × U(cell)`. A cell with too few ordinary balls uses the player's overall usual level. I propose matching on phase as the headline and adding innings as a second dimension only if enough players have the balls to support it. I will show the cell counts before fixing this.

**What Phase 3 will not do.** Shrinkage for small samples, confidence intervals, minimum-ball thresholds and how `P` and uplift combine into a ranking are all Phase 5.

**Reliability, stated early.** A wicket moves `v` by several runs on one ball, so per-ball values are noisy. With 200 pressure balls the standard error of `P` is roughly 15–20 runs per 100 balls (my estimate, assuming a wicket costs about 8 runs; the measured figure comes with the implementation). Only 74 batters and 80 bowlers have 200 or more pressure balls (142 and 149 have 100 or more). Shrinkage in Phase 5 is therefore essential, not optional.

**Exposure does not bias the level.** A player's share of pressure balls depends on how often their side chases (see the Phase 2b addendum). That changes how much evidence there is on a player, not the level of their score, because every ball is judged in its own state.

## 8. Companion metric: win probability added

`WPA` on a ball = win probability after the ball minus before it, from the batting side's view. The batter is credited with it on balls faced and the bowler with its negative.

- It is reported as a total and split into pressure and ordinary balls.
- It is **never weighted by LI and never capped**, because leverage is already inside it.
- It is used for storytelling ("added X wins under pressure") and as a Phase 6 cross-check: a ranking on LI-weighted run value should broadly agree with a ranking on pressure WPA.

## 9. Sanity checks I will show after implementing

1. **Zero by construction.** Mean `v` on held-out seasons, overall and by innings, phase, season, competition and pressure flag, each with its standard error.
2. **The raw gap closes.** The 145 v 138 strike-rate gap in §2 should become a `v` gap of about zero.
3. **Calibration** of all four expectation models, and their gain over the season × phase average.
4. **The exchange rate.** `c` by balls left × wickets in hand for the first innings, chase examples, its distribution, the share of balls using the fallback, and the proposed cap.
5. **Distributions.** Spread of `v` by phase and innings, and of player-level `P`, `U` and uplift.
6. **Face validity.** Correlation of batting `U` with plain strike rate and average, and of bowling `U` with economy, across players.
7. **WPA.** Mean of zero, and its player-level agreement with LI-weighted `v`.
8. **Unit tests** for the crediting rules in §3, for `c ≥ 0`, and for the zero-mean property on a synthetic table.

## 10. Known limitations

- Fielding, and blame for run-outs, cannot be measured from this data.
- A run is valued at one run wherever it is scored. In a few states (the last ball of a chase) the first run is worth more than the second.
- A batter gets no credit for rotating the strike to a better partner.
- Free hits are inferred from the previous delivery being a no-ball.

## 11. Decisions I need

1. **Value unit.** Run value with the DP exchange rate as the main metric, WPA as companion?
2. **Exchange rate.** The ratio definition in §5, with the fallback and a cap that I show you before using?
3. **Expectation features.** Include balls already faced by the striker and the free-hit flag? Being set is partly earned, so this is a judgement: I recommend including it, because otherwise finishers who walk in under pressure are marked against set batters.
4. **Crediting rules.** Non-striker run-outs charged to nobody, retired-out not charged, run-outs not credited to bowlers?
5. **Own baseline.** Matched on bowling phase, with innings added only if the cell counts allow?
6. **Reporting unit.** Runs per 100 balls for both batting and bowling?
