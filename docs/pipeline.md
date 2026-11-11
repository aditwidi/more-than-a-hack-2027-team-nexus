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

## Step 3 (candidates)

1. **Normalise ownership per team** (top-3 mean instead of a sum over slots) — attacks the measured
   home bias directly.
2. **Restart-aware dead ball** — use the set-piece `Type` and the kickoff anchors to place dead
   stretches, not just to add a bonus.
3. **Ball-state head** — visible-but-uncontrolled vs visible-and-controlled, using ball speed and
   nearest-player distance together (Peral et al. 2026 do this as a separate classification head).
4. **Tabular gradient boosting** on the feature matrix (Lang's AdaBoost configuration) as an
   emission model, trained on pseudo-labels — only if the rules plateau.
