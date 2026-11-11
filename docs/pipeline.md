# Pipeline — Step 1: features and label-free diagnostics

Step 1 makes no decisions. It turns the provided tracking, tags and homography into **evidence**,
and measures the evidence with statistics that need no labels. The decoder that turns evidence into
a label sequence is Step 2.

Design rationale, with citations: [references.md](references.md). Scoring contract:
[SUBMISSION.md](../SUBMISSION.md).

## Run it

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pipeline.features        # ~8 s -> artifacts/features.npz (182 x 199,319 float32, 145 MB)
python -m pipeline.diag --submit   # report + artifacts/diag.json + control submission
python evaluation/run_evaluation.py
python -m pytest tests/ -q         # 12 tests, <1 s, no match data needed
```

`pipeline.features` prints the matrix shape and the number of channels containing NaN (currently
**0** — every channel is defined on every frame, which the decoder can rely on).

## What Step 1 produces

### `artifacts/features.npz`
One float32 row per **video frame**, 199,319 rows, so row *i* is submission row *i*. 182 channels:

| group | examples | why |
|---|---|---|
| ball | `ball_present`, `ball_gap_prev`, `ball_gap_next`, `ball_fill_span`, `ball_x/y`, `ball_vx/vy`, `ball_edge_distance` | the ball is absent on 41% of graded frames; the *shape* of the absence is evidence (`D`), and interpolated positions carry a span so the decoder can discount them |
| ownership | `score_home/away` (soft count, exp(-d/4 m)), `score_margin`, `d_home/away`, `d_ratio`, `n_{home,away,diff}_r5/r10` | 5 nearest players reach 79.4% on the 3-class task vs 58.4% for ball-only (Borghesi et al. 2023) |
| team shape | `{home,away}_cx/cy/spread`, `d_centroid_*`, `home_player_share` | possession is a team property; `H` vs `A` is a geometry question |
| tags | `in_set_piece`, `tag_team` (+1/0/-1), 6 set-piece type flags, 13 code flags | every set piece is a restart; `Team` is `N/A` on all throw-ins and is handled as unknown, not defaulted |
| boundaries | `is_graded`, `since_half_start`, `to_half_end` | the halves, and the 2,934 all-`D` frames before the H2 kickoff |
| context (D4) | 7 channels x 8 log-spaced lags (0.4–64 s) x {lag, lead} = 112, plus 9 rolling means | Lang et al. 2022: position/velocity/acceleration at 15 multi-scale shifts, 92% on dead/alive with AdaBoost |
| quality | `reprojection_error` | homography fit confidence (tactical: median 0.24) |

Context is **acausal** (uses future frames). That is free here — submissions are scored offline —
and it is what makes boundary placement possible rather than causal-lagged.

### `artifacts/diag.json` — label-free measurements

Four families, none of which needs the ground truth:

1. **Ball visibility** — must match the source (56.9% overall, 59.3% graded). 150 absences, median
   268 frames (10.7 s), p99 2,618 frames (105 s), longest 5,537 (221 s). 64,570 graded frames sit in
   an absence longer than one second — the terrain of the challenge.
2. **Smart Tagging coverage** — 43,949 graded frames (27.7%) fall inside a set-piece interval. A
   dead-ball detector that never fires there is broken.
3. **Possession structure** — count, mean/median duration, dead share, home share, transitions per
   minute, and `short_possessions_lt_2s`. This is the family the ranking metric is built on, so a
   controller reporting 707 possessions in one half (median 1.44 s) is visibly wrong without any
   labels.
4. **Dead-ball proxies** — *recall*: share of set-piece frames predicted dead (control: 32%);
   *precision*: share of predicted-`D` frames explained by a set piece, a >1 s ball absence, or the
   pre-kickoff block (control: 99.8% — because the control's only `D` rule *is* ball absence).

### `evaluation/results/control_nearest.csv`

Nearest tracked team to the ball, `D` where the ball is invisible — the published baseline's rule,
rebuilt on the feature matrix so it cannot drift from the features and costs nothing to regenerate.
On the public window: **spellF1 0.0464, 95 possessions** (published floor `smooth50`: 0.3256 / 26).

## Verification performed

- **Tag parse against DATA.md.** Instance counts match exactly: PLAYERS IN THE BOX 97, BALL IN FINAL
  THIRD 93, BALL IN THE BOX 72, PROGRESSION 66, DEFENDING IN MIDDLE THIRD 66, SET PIECES 59; `Team`
  314/288/**36 N/A**; `Half` 320+318 = 638; `Type` 26/12/10/5/4/2. This also confirms the time base
  (`round(seconds * 25)`) and that nested `<label><group>/<text>` is the right parse — the naive
  `findtext("Team")` silently returns nothing and yields an all-unknown team tag.
- **Frame alignment.** `load_atd` asserts the raw file's ids are exactly 1…199,319 and stores them
  at 0…199,318. An off-by-one is a hard error, not a wrong score.
- **Ball coverage** 56.9% overall / 59.3% graded — matches DATA.md to three decimals.
- **12 unit tests**, including: interpolation is linear not last-known-held; velocity is zeroed
  across gaps (a re-acquisition is not motion); `shift` clamps at the edges instead of wrapping;
  rolling means skip NaN; an empty team yields the distance cap and zero margin; `H H D D H H` is
  two possessions, not one.

## What Step 1 deliberately does not do

- No labels, no smoothing, no thresholds — nothing here is tuned against the 5 public minutes.
- No video decode and no broadcast (see references.md D8).
- No per-player identity: the 46 slots are tracklets, so all evidence is collective.

## Step 2 (done): the decoder

`pipeline/emissions.py` scores each frame for `H`, `A` and `D` as log-odds; `pipeline/decode.py`
turns those scores into a possession sequence in two stages; `pipeline/submit.py` writes the file.

```bash
python -m pipeline.submit --out my_team.csv      # ~2 s, deterministic
python -m pipeline.ablate --write                # every variant -> evaluation/results/abl_*.csv
python evaluation/run_evaluation.py
```

### Emissions: two heads, not three classes

| head | evidence | prior justification |
|---|---|---|
| **dead** | ball gap length (0.55 log-odds/s, capped 8 s), ball beyond the touchline (`dead_out_of_play` 4.0), set-piece interval (0.9), long-gap bonus (1.2), player stillness (1.5), share of players moving (−1.2), loose ball (0.8), and a `dead_bias` of −1.5 | effective playing time in professional football is ~55–60 of 90 minutes, so ~35–40% dead; Lang et al. 2022 classify stoppages from player movement alone at 92% |
| **possession** | `score_margin` = Σexp(−d/4 m) over each team's slots, blended 0.2/0.4/0.4 over this frame, ±1 s and ±4 s, scaled by 1.8 | Borghesi et al. 2023: the five players nearest the ball reach 79.4% on the 3-class task vs 58.4% for ball-only |
| discount | ownership evidence multiplied by `exp(−fill_span/2 s)`, and by 0.35 when the ball is >25 m from everyone | in contested phases possession stays with the last team (CHALLENGE.md); an interpolated ball is a guess |

### Decoding: Viterbi, then structural cleanup

Switch costs are asymmetric in log-odds: **H↔A 6.0** (changing team *is* the possession event) versus
**H/A↔D 2.5** (stoppages are frequent and brief). Then runs shorter than the minimum dwell are
absorbed, with the priority order: same-label neighbour → blip inside a stoppage → short stoppage
inside a possession → longer neighbour. A dead stretch **never** merges the same team across it,
because `H H D D H H` is two possessions (CHALLENGE.md) and the metric is built on that.

### What the ablation says (`python -m pipeline.ablate`)

Label-free measurements, full match:

| variant | possessions | per half | mean s | dead share |
|---|---|---|---|---|
| `argmax` (no decoder) | 918 | 481/437 | 3.55 | 0.487 |
| no multi-scale context (weights 1/0/0) | 847 | 439/408 | 3.85 | 0.487 |
| no Viterbi (switch 0) → switch 2 → 4 → 6 | 918 → 578 → 522 → 495 | | 3.6 → 5.7 → 6.3 → 6.6 | 0.49 → 0.48 |
| no confidence discount | 495 | 255/240 | 6.62 | 0.484 |
| no distance gate | 495 | 255/240 | 6.62 | 0.484 |
| **final** (switch 6, dwell 2 s, dead bias −1.5) | **436** | 214/222 | **8.07** | **0.447** |

Read honestly: **multi-scale context and the switch penalty do the work** (918 → 495 possessions);
the confidence discount and distance gate are nearly neutral once context is in place, and are kept
because they are cheap and defensible, not because the ablation demands them. The dead-ball features
are what raised `D` recall: predicted-dead share of set-piece frames 0.32 → 0.47.

### Two features worth more than their cost

Measured on the public window, as a check that they mean what we think (not as a fitting target):

| signal | dead frames | live frames |
|---|---|---|
| ball beyond the touchline/goal line | **27.6%** | **0.0%** |
| mean tracked player speed | 1.43 m/s | 2.38 m/s |
| share of players above 3 m/s | 8.7% | 30.6% |
| nearest player > 5 m from the ball | 35.3% | 18.1% |

The first is nearly pure evidence and was invisible in the original feature set: a ball outside the
pitch coordinates is out of play, always.

### Scores on the public window (a consistency check, not a target)

| submission | macroF1 | spellF1 | possessions (truth 17) | mean duration (truth 14.2 s) |
|---|---|---|---|---|
| `baseline_smooth50` (published floor) | 0.6207 | 0.3256 | 26 | 10.8 |
| `baseline_smooth125` | 0.5812 | **0.4242** | 16 | 18.5 |
| **this pipeline** | **0.6400** | 0.3889 | 19 | 13.6 |

It beats the published floor on both metrics and lands between the two smoothing baselines on
`spellF1` — while, unlike `smooth125`, staying inside the label-free structure priors. Per
`EVALUATION.md`, the public window cannot rank these: `smooth125` is the best of the published
baselines on 17 possessions and the *worse* of the two on the full match.

### Known weaknesses (measured, not guessed)

| weakness | evidence |
|---|---|
| home bias in live play | predicted alive share on H = 0.71 vs truth 0.57 in the public window; `A` recall 0.55 vs the baseline's 0.77. The track slots are 24 home / 22 away, so an unnormalised Σexp(−d) is biased toward home; a top-k mean per team is the obvious fix. |
| dead ball with a visible ball | `D` recall 0.43; 65% of dead frames in the window have a visible ball and we catch ~10% of those. Restart timing (throw-in, free kick) is the missing evidence. |
| possession length still short | 8.1 s mean vs the 8–15 s prior, at the lower edge. More evidence, not more smoothing, is the lever — the switch penalty saturates by 8. |
| boundary latency | mean −6.3 frames at transitions; the first frames of a possession are systematically missed. |

## Step 3 (in progress): normalise ownership, fix the tag prior

### 1. Per-team top-k ownership — done

`score_margin` summed `exp(-d/4)` over *every* slot, so a team with more slots won ties it should
have lost: this match has **24 home slots and 22 away**, and a duplicated track 8 m from the play
adds 0.14 to that team's score. Replaced with `margin_top3`, the mean weight of the **k = 3 nearest
players per team**, which is invariant to slot count and closer to Borghesi et al.'s "five closest
players" representation.

Measured effect, on the public window as a semantics check: the *sign* of the raw margin agreed with
the truth's home/alive share 58.4% of the time against a truth of 57.3%; the normalised one agreed
**57.4%** of the time. Predicted alive home share fell from 0.709 to 0.638 before the tag fix below
(and to 0.619 after), against a truth of 0.573. Possession count in the window 22 → 18.

Ablation, full match: `alb_unnormalised_margin` leaves 529 possessions against 365 here.

### 2. The Smart Tagging prior was inverted and overwritten — fixed

Two bugs, both found by measuring the prior instead of trusting it:

1. **Defending codes are tagged with the team *without* the ball** (and Defensive Transition with the
team that just lost it), so using the raw `Team` sign pushed possession the wrong way for roughly a
third of the match. `io.TAG_POSSESSION_SIGN` now carries a sign per code.
2. **Each phase is recorded twice** — once with the attacking team's code, once with the defending
team's code over the same interval (`PROGRESSION` and `DEFENDING IN MIDDLE THIRD` have identical
frame counts, as do `ATTACKING`/`DEFENSIVE TRANSITION`). The old code assigned a per-frame team tag
for each instance in turn, so the last instance written won: the tag was effectively random within
a phase. `tag_possession` now accumulates `sign(code) x team` over both instances and averages.

After the fix the prior is informative: **P(home | prior home) = 0.769** against a base rate of
0.573, and P(home | prior away) = 0.435. It is applied at weight 0.35, deliberately subdominant to
geometry — at 0.6 team accuracy on live frames fell 0.810 → 0.734 while the prior took over.

### Where this leaves the pipeline

| variant | macroF1 | spellF1 | possessions (truth 17) | duration (truth 14.2 s) | share err |
|---|---|---|---|---|---|
| `baseline_smooth50` (floor) | 0.6207 | 0.3256 | 26 | 10.8 | 0.048 |
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 | 18.5 | 0.052 |
| `nexus_v4` (Step 2) | 0.6439 | 0.4343 | 18 | 14.5 | 0.119 |
| **`nexus_v5` (Step 3)** | **0.6511** | **0.4471** | **17** | 15.2 | **0.047** |

D-class F1 0.48 against 0.32 (smooth50) and 0.15 (smooth125). Label-free full match: 365
possessions, mean 9.76 s, dead share 0.446, home share 0.587.

### Still open

| weakness | status |
|---|---|
| `D` recall ≈ 0.41 with a visible ball | restart timing (throw-in / free kick) is not yet used as *placement*, only as a bonus |
| `A` recall 0.65 vs the baseline's 0.77 | away possessions with a missing ball are still absorbed into the previous state |
| boundary latency −2.8 frames (window) | improved from −6.3, still early |
| home share 0.619 vs 0.573 (window) | 0.587 label-free; within the noise of a 5-minute sample, but worth watching |

## Step 4: restart placement (negative) and ball-stationary duration (positive)

### 1. Restart-aware dead ball — tried, measured, rejected

The hypothesis: a tagged `SET PIECES` interval spans the whole restart sequence, so its *shape* can
place dead ball. The prior is real — **P(dead | interval start) = 0.571** against a 0.196 base rate —
but using it as a dead-ball booster made everything worse:

| `dead_restart` weight | D recall | D precision | team accuracy | dead share |
|---|---|---|---|---|
| 0.0 | 0.41 | 0.58 | 0.773 | 0.446 |
| 2.5 | 0.43 | 0.52 | 0.747 | 0.493 |
| 4.0 | 0.49 | 0.47 | 0.708 | 0.523 |

It buys recall by claiming live frames: 43% of the interval start is *not* dead, because play
resumes inside the interval and nothing tells us where the restart action sits (the Kick Off (Start)
instance covers 443.88–448.88 s for a kickoff struck at 447.44 s, i.e. 3.5 s in, and that offset is
not constant across set-piece types). The weight is therefore **0.0** by default, the feature and the
kernel stay in the code, and the negative result is recorded here rather than deleted. What would
make restarts usable is knowing *when the ball was struck* — a ball-state head, not a prior.

### 2. Ball-stationary duration — kept

The instantaneous channels could not express "the ball has not moved for six seconds", which is what
a free kick or a throw-in being set up looks like. A run-length feature (`ball_still_frames`) and its
combination with slow players (`dead_still_combo`) added the missing evidence:

- `P(dead | ball still > 2 s and players slow) = 0.494` against a 0.196 base, covering 40% of dead
  frames — exactly the population the earlier model missed (dead with a *visible* ball).
- At weight 4.0: **D recall 0.41 → 0.56**, **D precision 0.58 → 0.68**, team accuracy 0.773 → 0.797,
  dead share 0.446 → 0.475.

Weights 3/4/5/6 gave spellF1 0.4343 / 0.4444 / 0.4111 / 0.3568 and macroF1 0.6699 / 0.7205 / 0.7548 /
0.7504. 4.0 is where precision and recall both improved and the dead share stays closest to the
football prior; above that the decoder starts calling live frames dead and the structure degrades.

### 3. Minimum possession 2 s → 3 s

A post-pass on the finished sequence, chosen on the same CHALLENGE.md definition (controlled
possession = the ball kept a few seconds) and on label-free structure: 398 possessions over the match
at 8.45 s mean, against 401 at 8.38 s for 2 s. Both metrics agree on 3 s, which is unusual enough to
be worth noting.

### Result

| submission | macroF1 | spellF1 | possessions (truth 17) | duration (truth 14.2 s) | share err | D F1 |
|---|---|---|---|---|---|---|
| `baseline_smooth50` (floor) | 0.6207 | 0.3256 | 26 | 10.8 | 0.048 | 0.32 |
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 | 18.5 | 0.052 | 0.15 |
| Step 3 | 0.6511 | 0.4471 | 17 | 15.2 | 0.047 | 0.48 |
| **Step 4 (`nexus_v7`)** | **0.7335** | **0.4571** | 18 | 13.8 | 0.024 | **0.68** |

Label-free, full match: 398 possessions (183 / 215 per half), mean 8.45 s, dead share 0.476, home
share 0.581.

### Still open

| weakness | status |
|---|---|
| `A` recall | 0.65 vs the baseline's 0.77; away possessions with a missing ball are still absorbed into the previous state |
| home share 0.61 / 0.55 per half | label-free 0.581; the public window's truth is 0.573, so the residual is small but unexplained |
| boundary latency −3 to −7 frames | transitions are placed slightly early; a restart-timing signal would fix both this and the rejected restart prior |
| dead share 0.476 | above the 35–40% football prior; the still-ball signal is doing real work but may be over-firing in the second half (0.494) |

## Step 5: recalibrated dead head, formation fallback, discounts removed

### 1. The gap→dead weight was too strong

The ground truth is derived from **corrected** tracking, which knows the ball is in play while ours
has lost it. Measured on the public window: 391 truth-alive frames had a **median ball gap of 5.6 s**,
and the old weight called 98.7% of them dead. Lowering `dead_per_second` 0.55 → 0.40 and the long-gap
bonus 1.2 → 0.8 raised **D precision 0.69 → 0.75 with recall unchanged** — fewer false stoppages, not
fewer real ones.

### 2. Formation fallback, entering with an inverted sign

The one population the model could not reach was missing-ball frames. The team **centroid** solves
it — but not in the direction you would guess. On ball-absent frames in the public window, the team
whose centroid was *nearer* the ball was the **defending** team 384 times out of 384: the side out of
possession is compact around the ball while the side in possession spreads. And with the ball
visible the relation flips again (it inverts), so the term is gated on `ball_present == 0`.

Adding it with the natural sign *hurt* (A recall 0.756 → 0.721). With the sign corrected:

| `shape_scale` | A recall | team accuracy | boundary latency |
|---|---|---|---|
| 0.0 | 0.756 | 0.813 | −6.6 frames |
| −0.10 | 0.765 | 0.817 | **−1.1 frames** |
| −0.25 | 0.809 | 0.836 | −4.3 frames |

−0.10 is used: it gains recall and improves boundary placement sharply without inflating the
possession count the way −0.25 does.

### 3. The interpolation discount was made obsolete by #2

`possession_confidence_s` (discounting ownership evidence by `exp(-span/2 s)`) was introduced in
Step 2 when the interpolated ball was the only read on a missing ball. With the formation fallback
in place it is strictly harmful, and it is now **off** (`0.0`, gate factor `1.0`): team accuracy
0.817 → 0.835, A recall 0.765 → 0.807, share error 0.011 → 0.002, label-free structure unchanged
(418 vs 422 possessions). It stays as a documented knob because the ablation reports it.

### 4. Boundary refinement: measured, neutral, left off

`refine_boundaries()` re-places each transition at the maximum-likelihood change point of the
instantaneous logits. Structure is unchanged at ±25/50/75/100 frames (418 possessions in every
case); it moved 40 frames and nothing else. Kept in the code, tested, **default 0** — shipping a
mechanism that claims to help and measurably does not would be worse than shipping nothing.

### 5. A measurement warning worth recording

`abl_no_confidence` and `abl_final` are now the **same configuration**: the discount is off by
default, and the ablation sets it to `1e6`, which is `exp(-span/1e6) ≈ 0.99999` rather than exactly
1.0. That 1e-5 difference is enough to break ties differently in the Viterbi argmax: **568 frames
change, and the public spellF1 moves 0.4800 → 0.4947**. So a difference of 0.015 here is numerical,
not modelling. It is a concrete instance of the warning in `EVALUATION.md`: on a 17-possession
window, treat anything under ~0.04 as noise, and do not read the third decimal place at all.

### Result

| submission | macroF1 | spellF1 | possessions (truth 17) | duration (truth 14.2 s) | share err | D F1 |
|---|---|---|---|---|---|---|
| `baseline_smooth50` (floor) | 0.6207 | 0.3256 | 26 | 10.8 | 0.048 | 0.32 |
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 | 18.5 | 0.052 | 0.15 |
| Step 4 | 0.7335 | 0.4571 | 18 | 13.8 | 0.024 | 0.68 |
| **Step 5** | **0.7666** | **0.4800** | 18 | 14.3 | **0.002** | **0.69** |

Predicted home share of live play **0.574 against a truth of 0.573**; per-class F1 H 0.825 / A 0.781 /
D 0.694; boundary latency −1.1 frames; fragmentation 1.29. Label-free full match: 418 possessions
(195 / 223 per half), mean 8.32 s, dead 0.457, home share 0.569.

Still losing: 10 spurious and 9 missed possessions at IoU 0.5, and only 5 of 20 true transitions are
matched within ±25 frames — the remaining gap is in deciding *which* transitions exist, not where
they sit.

### Ablation summary (label-free, full match)

| variant | possessions | mean s | dead |
|---|---|---|---|
| argmax, no decoder | 760 | 4.54 | 0.457 |
| no multi-scale context | 721 | 4.83 | 0.453 |
| unnormalised margin | 591 | 5.71 | 0.468 |
| raw (unsigned) tag prior | 519 | 6.60 | 0.461 |
| no tag prior | 481 | 7.08 | 0.464 |
| switch penalty 2 / 4 / **6** / 10 | 492 / 454 / **421** / 421 | 7.06 / 7.64 / **8.23** / 8.23 | 0.456 |
| no still-ball duration | 383 | 9.61 | 0.427 |
| restart prior at weight 2.5 (rejected) | 408 | 7.61 | 0.516 |
| **final** | **418** | **8.32** | **0.457** |

## Step 6 (candidates)

1. **Transition existence** — 5 of 20 true transitions matched. The decoder still invents transitions
   inside possessions; a duration-aware model (HSMM instead of a constant switch penalty) is the
   principled fix.
2. **Restart timing from ball-motion onset** — the frame a stationary ball starts moving is the
   restart action; that would place dead ball instead of boosting it (Step 4's negative result) and
   would fix the remaining boundary error.
3. **Tabular gradient boosting** on the feature matrix, trained on pseudo-labels, using the same
   features — the only remaining route that could beat hand-set priors.
