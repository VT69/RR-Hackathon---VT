# Phase 4a — Fair comparisons: proposal

No code yet. This sets out how players who faced different opposition, in different competitions, in different roles and with different amounts of evidence are made comparable. Decisions needed are in §10.

## 1. The proposal in five lines

1. **Opposition and competition strength are adjusted together, ball by ball,** with one "who faced whom" model: every ball's run value is explained by the batter's skill and the bowler's skill. A player is then credited for the quality of the opponent on each ball.
2. **IPL and T20I are pooled into one ranking.** The model puts them on a common scale through the 182 batters and 143 bowlers who play in both. Associate matches stay in, adjusted like any other opposition.
3. **Role is handled by the match state already in run value.** No extra adjustment for batting position; a position-relative version is a Phase 6 check.
4. **Eligibility: at least 100 pressure balls** (faced or bowled) **and pressure balls in at least 10 matches.** That gives about 140 batters and 150 bowlers.
5. **Shrinkage happens once, at the end** (Phase 5), on the adjusted values, with empirical-Bayes weights measured from the data.

## 2. What run value already adjusts for, and what it does not

| Source of unfairness | Handled by Phase 3? | Where it is handled |
|---|---|---|
| Match state (overs, wickets, target, chase rate) | Yes | Expectations in run value |
| Era and competition scoring level | Yes | Zero-mean within each competition-season |
| Being set or new at the crease, free hits | Yes | Expectation features |
| **Quality of the opponents faced** | **No** | §4 |
| **Strength of the competition itself** (IPL v T20I, top teams v others) | **No** | §4 and §5 |
| Batting position, bowling phase | Partly (through the state) | §6 |
| Different amounts of evidence | No | §7 (eligibility) and Phase 5 (shrinkage) |

## 3. Evidence that opposition and competition matter

"Zero-mean within each competition-season" means a T20I player is measured against the T20I average, and an IPL player against the IPL average. If those averages differ in strength, the same performance scores differently.

**The cleanest test is the same player in two settings.** Run value per 100 balls, same players, at least 150 balls in each setting:

| Same players | Comparison | Difference | Standard error | Players |
|---|---|---|---|---|
| Batters | v T20I top-8 bowling, minus v IPL bowling | **+9.5** | 2.7 | 59 |
| Batters | v other full members' bowling, minus v top-8 | +1.2 | 3.8 | 43 |
| Bowlers | v T20I top-8 batting, minus v IPL batting | −2.9 | 2.7 | 66 |
| Bowlers | v other full members' batting, minus v top-8 | **+17.0** | 3.6 | 45 |

"Top-8" is India, Pakistan, Australia, England, New Zealand, South Africa, West Indies, Sri Lanka. "Other full members" is Bangladesh, Afghanistan, Ireland, Zimbabwe. Associates are too rare for a same-player comparison (4% of balls).

What this shows:

- **IPL bowling is harder to bat against than top-8 international bowling**: the same batters score 9.5 runs per 100 balls better, relative to the league, in T20Is. A batter who plays mostly T20Is is flattered against one who plays mostly IPL.
- **Batting by the four other full members is much easier to bowl at**: the same bowlers are 17 runs per 100 balls better against it.
- These gaps are as large as the gap between a top-10 and an average player (§6 of the Phase 3 report).
- Overall, in the comparison of the same players in IPL and in all T20Is, batters are 11.6 per 100 better in T20Is (se 2.3) and bowlers 4.1 better (se 3.0).

**Your test cases.**

| Bowler | Overall value / 100 | Share of balls v other full members or associates | Share v top-8 | Share in IPL |
|---|---|---|---|---|
| Blessing Muzarabani (#1) | +24.9 | 51% | 43% | 5% |
| Mohammad Nawaz | +20.6 | 33% | 66% | 0% |
| Mahedi Hasan | +15.4 | 41% | 59% | 0% |
| Jasprit Bumrah (for contrast) | +25.8 | 9% | 20% | 71% |
| Chris Jordan | −26.4 | 6% | 56% | 38% |
| Sam Curran | −16.2 | 6% | 51% | 43% |
| Anrich Nortje | −18.9 | 0% | 31% | 69% |
| Alzarri Joseph | −15.9 | 8% | 59% | 33% |
| Romario Shepherd | −21.5 | 13% | 64% | 23% |

The three you flagged bowl a large share at the weaker sides. The five death bowlers bowl almost entirely at top-8 and IPL batting. Among eligible players, 38 batters and 41 bowlers have more than a quarter of their balls against other full members or associates.

## 4. Proposal: one "who faced whom" model

### 4.1 The idea

Every ball has a batter and a bowler. Run value on that ball is treated as:

> value = batter's skill − bowler's skill + luck

Fitted over all 193,501 balls at once, this separates the two. A bowler who bowls mostly at weak batters gets less credit for the same figures, because those batters' skill estimates are low, measured from all their other balls against everyone else.

There are two fits, one per value:

- **Batting value** explained by a batter effect plus a bowler effect.
- **Bowling value** explained by a bowler effect plus a batter effect.

Two fits are needed because batting and bowling value are not exact mirrors (byes, run-outs and the non-striker's share are credited differently).

### 4.2 How it is used: an adjustment per ball

The fitted model is not used as the rating. It supplies only the **opponent's** skill on each ball:

> adjusted value on a ball = run value + skill of the opponent on that ball

(For a batter, facing a bowler who usually concedes 10 runs per 100 fewer than expected adds back 0.10 runs to that ball; facing a weak bowler subtracts.)

Why adjust per ball instead of using the model's own player ratings:

- **The pressure split survives.** Pressure value, usual value, uplift and consistency in Phase 5 are all averages of the same adjusted balls.
- **It stays explainable.** Each ball's adjustment is one number: who was at the other end.
- **Shrinkage happens once.** The model shrinks the opponents' skills; the player's own rating is shrunk once, in Phase 5. Nothing is shrunk twice.
- **Leverage still enters once,** at the pressure stage. The model is fitted on unweighted run value.

### 4.3 Technical choices

| Choice | Proposal | Why |
|---|---|---|
| Method | Ridge regression (each player's skill is pulled towards zero by a penalty) | It is the standard way to fit thousands of player effects without wild values for players with few balls. It is the same as treating skills as random effects |
| Penalty strength | Chosen by prediction on held-out matches (odd v even matches) | Measured from the data, not set by hand |
| Data | All scored balls, both innings, unweighted | Opponent quality is a property of the opponent, not of the moment |
| Common scale | Ball-weighted average adjusted value = 0 over all balls | One neutral reference point for both competitions |
| What links IPL and T20I | The 182 batters and 143 bowlers who play both; they face 49% of T20I balls | Enough overlap for the model to learn the strength difference from the data |

### 4.4 How I will check it worked

1. **The same-player gaps close.** The +9.5 (batters, top-8 v IPL) and +17.0 (bowlers, other full members v top-8) should fall towards zero after adjustment. This is the main test, and it uses no information the model was built to fit directly.
2. **It predicts better.** The model with opponent effects should predict run value on held-out matches better than one without them.
3. **Your test cases move as cricket logic expects.** Muzarabani, Nawaz and Mahedi Hasan should fall; the five death bowlers should rise. If they do not, I will report it, not tune it.
4. **Size of the adjustment.** Distribution of each eligible player's total adjustment, and which players move most.

### 4.5 Alternatives considered

| Option | Verdict |
|---|---|
| **A. Who-faced-whom model, per-ball adjustment** | Recommended |
| B. Fixed offsets by opposition tier (for example +17 for balls against the four other full members) | Simple, but coarse: treats all of Zimbabwe's batters, or all of IPL, as one strength. Kept as a Phase 6 check |
| C. Opponent team-season strength | Ignores who actually batted or bowled; a team's strength changes with its line-up |
| D. Separate rankings for IPL and T20I | Does not answer the brief, which asks for one top 10 per role |
| E. No adjustment | §3 shows the bias is as big as the differences being ranked |

## 5. IPL, T20I and associates

- **Pool both competitions into one ranking,** on the common scale from §4.
- **Associates: include** their balls, adjusted like any other opposition. They are 4% of all balls and 0% for the median eligible player. A ranking with associate balls removed is a Phase 6 check. This settles the open point in D10.
- **Every ranking output shows each player's mix**: share of balls in IPL, against top-8, against other full members and against associates. A reader can see where a rating comes from.
- **Knockouts:** no change (records reported separately, ×1.25 as a Phase 6 check).

## 6. Role

**Batting position.** After the Phase 3 calibration, mean batting value by position is about −4 per 100 balls for openers, +4 for number 4, and strongly negative for 9–10. The opener gap sits in the runs part, among openers who are set and still batting in the middle overs (Phase 3 report, addendum).

Two ways to treat this:

| Option | What it means | My view |
|---|---|---|
| **Keep the state-based baseline** (no position adjustment) | An opener is compared with any batter in the same match state | **Recommended.** The state already carries the role (overs, wickets, how set the batter is). Adjusting for position would declare every position equal on average and remove a difference that may be real |
| Position-relative baseline | An opener is compared with other openers | A Phase 6 check, to show how much the batting top 10 depends on this choice |

**Bowling phase.** Death bowlers face stronger batters (§3). That is opposition, so the §4 adjustment handles it. No separate adjustment for phase.

**Pace or spin.** Not adjusted. The match state already carries the situations each type is used in.

## 7. Eligibility

Pressure balls per player (balls faced as striker for batters, balls bowled for bowlers):

| Minimum pressure balls | Batters | Their share of all batting pressure balls | Bowlers | Their share of bowling pressure balls |
|---|---|---|---|---|
| 60 | 195 | 83% | 200 | 86% |
| **100** | **142** | **74%** | **149** | **78%** |
| 150 | 103 | 64% | 107 | 67% |
| 200 | 74 | 53% | 80 | 57% |
| 300 | 30 | 30% | 46 | 40% |

- Median batter: 12 pressure balls; 90th percentile 183. Median bowler: 29; 90th percentile 250.
- At 100 pressure balls the median eligible batter has pressure balls in 22 matches and the median bowler in 20.

**Proposal: at least 100 pressure balls and pressure balls in at least 10 matches.**

- **Why 100 and not more.** A higher bar mostly removes players from teams that bat first more often (Phase 2b §5.4), not weaker players. Shrinkage in Phase 5 handles the extra noise of smaller samples, and the report states each player's evidence.
- **Why 10 matches.** So that a rating cannot rest on two or three extraordinary games. I will report how many players the second condition removes.
- **Players below the bar** appear in the outputs as "insufficient pressure evidence", with their numbers, but are not ranked. If a name on the Unstop dropdown falls below the bar, it is listed that way, not dropped silently.
- Thresholds go in `config.yaml`. The top 10 at 60 and 150 pressure balls is a Phase 6 check.

## 8. Shrinkage (design now, applied in Phase 5)

**Method.** Empirical Bayes, one component at a time (pressure value, usual value, uplift, consistency):

> shrunk value = observed value × reliability, where reliability = real spread² ÷ (real spread² + standard error²)

- **Real spread** is measured across eligible players as observed variance minus average squared standard error (the test already used in Phase 3).
- **Standard errors** come from resampling whole matches, not balls, because balls within a match are related.
- **Prior mean** is zero (league average after the opposition adjustment).

**What the Phase 3 data already say** (runs per 100 balls, after the wicket-cost calibration; implied real spread across players, by minimum pressure balls):

| | ≥ 100 | ≥ 200 | ≥ 300 |
|---|---|---|---|
| Batters: usual value | 23.4 | 8.5 | 9.5 |
| Batters: pressure value | 12.8 | 0.6 | 6.4 |
| Batters: uplift | 24.7 | **0.0** | 2.0 |
| Bowlers: usual value | 8.0 | 6.4 | 6.5 |
| Bowlers: pressure value | 7.7 | 10.7 | 8.6 |
| Bowlers: uplift | **0.0** | 5.6 | **0.0** |

These estimates jump around because each rests on 30–150 players, and the ≥ 100 column includes players whose standard errors are themselves uncertain. Phase 5 replaces them with the split-half test you specified. Two readings already look safe:

- **Uplift has little or no measurable spread for bowlers either**, not only for batters.
- **Pressure value carries real spread for bowlers** (about 8–11 runs per 100). For batters it is unstable and must be judged on the split-half test.

**Guarding against rewarding volume** (your Phase 5 condition). Shrinkage towards zero lets players with many pressure balls reach the extremes. Phase 5 will report:

- the correlation of the final rating with pressure-ball count, compared with the same correlation for unshrunk values;
- the top 10 if every eligible player is cut to the same number of pressure balls (a random subsample, repeated);
- how many top-10 players are from the teams that chase most often.

**An option for Phase 5, flagged now.** If uplift really has no spread, the best estimate of a player's pressure performance is their own usual level plus a heavily shrunk uplift. That would make the pressure ranking close to an overall ranking. Whether that is the honest answer is a Phase 5 decision, made on the split-half evidence.

## 9. Sanity checks I will show after implementing

1. The same-player gaps from §3, before and after.
2. Held-out prediction with and without opponent effects.
3. Your test cases: Muzarabani, Nawaz, Mahedi Hasan, and the five death bowlers, before and after.
4. Size of each eligible player's adjustment, and who moves most.
5. The overall top and bottom 10 again, after adjustment (still no pressure layer).
6. Mean adjusted value by competition, opposition tier and batting position.
7. The eligible pool: counts, competition mix, and any dropdown names below the bar.

## 10. Decisions needed

| # | Question | My recommendation |
|---|---|---|
| 1 | Opposition and competition strength | Who-faced-whom ridge model, used as a per-ball opponent adjustment (option A); tier offsets as a Phase 6 check |
| 2 | Penalty strength of that model | Chosen by prediction on held-out matches |
| 3 | IPL and T20I | One pooled ranking on a common scale; each player's competition and opposition mix shown |
| 4 | Associates | Include, adjusted; exclusion as a Phase 6 check |
| 5 | Batting position | No position adjustment; position-relative baseline as a Phase 6 check |
| 6 | Eligibility | ≥ 100 pressure balls and pressure balls in ≥ 10 matches; 60 and 150 as Phase 6 checks |
| 7 | Shrinkage | Empirical Bayes per component, match-level standard errors, prior mean zero; volume checks as listed in §8 |
