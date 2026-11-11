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

## Step 2 (next): the decoder

Emissions from these features (dead/alive head separate from the H/A head), then structured
decoding — Viterbi/HSMM over `{H, A, D}` with a switch penalty and minimum dwell. Judged by the same
label-free report plus the public window **only for correctness** (`--bootstrap 1000` before
believing any delta), against the two controls already on the board.
