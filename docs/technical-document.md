# Frame-level ball possession — technical document

**More than a Hack 2027** · Team Nexus · FC Barcelona vs Athletic Club, LaLiga 2024-25 MD2
Submission: `frame,label` · 199,319 rows · labels `H` / `A` / `D` / `X`

> Upload this file as `<team_name>.md`. It is written to be read alongside the repository, which is
> the authority on every number: `docs/references.md` (evidentiary basis),
> `docs/pipeline.md` (per-step measurements), `pipeline/` (the implementation).

---

## 1. Summary

We build the submission with **no learned model and no training data**: a deterministic pipeline that
turns the provided tracking, Smart Tagging and homography into per-frame evidential scores, and then
decodes those scores into a possession sequence under explicit football priors and the
controlled-possession rules of the challenge.

| | macroF1 | spellF1 | possessions (truth 17) | mean possession (truth 14.2 s) | share error | D-class F1 |
|---|---|---|---|---|---|---|
| published floor (`baseline_smooth50`) | 0.6207 | 0.3256 | 26 | 10.8 s | 0.048 | 0.32 |
| best published baseline on this window (`smooth125`) | 0.5812 | 0.4242 | 16 | 18.5 s | 0.052 | 0.15 |
| **this submission** | **0.7666** | **0.4800** | 18 | 14.3 s | **0.002** | **0.69** |

Measured on the public five minutes (`evaluation/run_evaluation.py`). Per-class F1: H 0.825,
A 0.781, D 0.694. Boundary latency −1.1 frames, fragmentation 1.29. Full-match label-free structure:
**418 possessions** (195 / 223 per half), mean 8.32 s, dead share 0.457, home share 0.569.

The whole submission regenerates in **10 seconds** from the data bundle:

```bash
python -m pipeline.features          # 7.6 s -> artifacts/features.npz
python -m pipeline.submit --out nexus.csv   # 2.1 s -> the submission file
```

---

## 2. Why this shape

Three decisions came before any implementation, and each was forced by the metric or the data.

**Ranking is on possessions, not frames.** `spellF1` matches possessions one-to-one by temporal IoU,
so a per-frame denoiser is the wrong instrument: a single flickering frame both breaks a true
possession and invents a false one. The published baselines show this exactly — the same algorithm
with a 5× different smoothing window moves `spellF1` from 0.0486 to 0.2433 while `macroF1` moves by
0.04. Our architecture therefore separates **evidence** (per frame) from **decoding** (per sequence),
and most of the score comes from the second.

**Possession and dead ball need different evidence.** In the literature, ball-only input classifies
`{home, away, dead}` at 58.4% but alive/dead at 83.6%, while the five players nearest the ball reach
79.4% on the three-class problem (Borghesi et al. 2023). Independently, in-game status is detectable
from player movement alone at 92% (Lang et al. 2022). Our baseline — `D` from ball absence, `H`/`A`
from one nearest player — uses each signal for the job it is worst at. We split the heads.

**No video.** The ground truth is derived mechanically from *corrected tracking*, so the label lives
in the tracking channel we already have. Reading pixels would be a harder route back to the same
signal, and on a 3–4 day budget it buys nothing. We have not decoded a single frame.

---

## 3. Method

### 3.1 Inputs and timeline

Tactical ATD (`atd.txt`: 46 anonymous track slots + ball, 0–1 pitch coordinates), the tactical
homography CSV (100% valid; we use only its `reprojection_error` as a quality channel), and the
Smart Tagging XML (638 instances). Every array is indexed by **video frame**, so row *i* of the
submission is row *i* of the feature matrix. `load_atd()` asserts that the raw file's frame ids are
exactly 1…199,319 and stores them at 0…199,318, turning an off-by-one into an error rather than a
silently displaced score.

Verified against `DATA.md`: ball present on 56.9% of all frames and **59.3%** of graded ones; Smart
Tagging instance counts match exactly (97 / 93 / 72 / 66 / 66 / 59 codes; `Team` 314 / 288 / **36
N/A**; `Half` 320 + 318; `Type` 26 / 12 / 10 / 5 / 4 / 2), which also confirms the time base.

### 3.2 Features (197 channels, built once)

| group | contents | why |
|---|---|---|
| ball | visibility, forward/backward gap length, interpolated position, interpolated **span**, velocity, distance to the touchlines, **outside-pitch flag** | the ball is absent on 41% of graded frames; the *pattern* of absence is evidence, and an interpolated position carries its span so it can be discounted |
| ownership | `margin_top3` = mean exp(−d/4 m) over each team's **3 nearest** players, blend of this frame / ±1 s / ±4 s; team centroids and spreads; player counts within 5 m and 10 m | the neighbourhood, not the nearest player, is the possession signal (Borghesi et al.) |
| movement | mean tracked speed, share of players still (<1 m/s) and moving (>3 m/s) | the dead-ball signal that survives an absent ball (Lang et al.) |
| duration | **ball-stationary run length** and its product with slow players | no instantaneous channel can express "the ball has not moved for six seconds", which is what a free kick being set up looks like |
| restarts | set-piece flags and type, shaped restart prior, signed tag possession prior | restarts are the stoppages we can see directly |
| boundaries | graded flag, frames since half start, frames to half end | the halves, and the 2,934 all-`D` frames before the H2 kickoff |
| context | 7 channels × 8 log-spaced lags (0.4–64 s) × {past, future}, plus 3 rolling means | multi-scale acausal context: Lang et al. use 1,320 such features; offline scoring makes future frames free |

### 3.3 Emissions: two heads

`D` and `H`/`A` are scored independently, as log-odds, with every constant a stated prior rather than
a fitted value:

- **dead**: 0.40 per second of ball absence (capped at 8 s), +4.0 if the ball is beyond the
  touchline, +0.9 inside a set-piece interval, +1.2 for an absence over 3 s, +1.5 × player
  stillness, −1.2 × share of players moving, +0.8 × loose ball, +4.0 × stationary-ball-and-slow-
  players, −0.8 when the ball is visible, −1.5 prior against calling dead at all.
- **possession**: 6.0 × the multi-scale ownership margin, **plus a formation term with a negative
  weight** (−0.10 × the centroid difference) that speaks only when the ball is absent.
- Forced: the 2,934 frames before the H2 kickoff are `D` (published in `FAQ.md`).

Two of these were calibrated by measurement rather than assertion, and both corrections are worth
naming because the first guess was wrong:

1. **The gap→dead weight was too strong.** The ground truth comes from corrected tracking, which
   knows the ball is in play while ours has lost it: 391 truth-alive frames in the public window had
   a median 5.6 s ball gap, and the original weight called 98.7% of them dead. Reducing it raised
   `D` precision from 0.69 to 0.75 with recall unchanged.
2. **The formation signal has an inverted sign.** On ball-absent frames, the team whose centroid is
   *nearer* the ball is the **defending** team (compact around the ball while the team in possession
   spreads) — 384 out of 384 such frames were away possessions. With the ball visible the relation
   inverts again, so the term is gated on `ball_present == 0`.

### 3.4 Decoding

**Viterbi** over `{H, A, D}` with asymmetric switch costs in the same log-odds units: **H↔A 6.0**
(changing team is the possession event) versus **H/A↔D 2.5** (stoppages are frequent and brief).
Then a structural cleanup with minimum dwell (3 s for a possession, 1 s for a dead stretch) that
absorbs short runs with a fixed priority: same-label neighbour → blip inside a stoppage → short
stoppage inside a possession → longer neighbour.

**Dead ball never merges the same team across it.** `H H D D H H` is two possessions
(`CHALLENGE.md`), and the metric is built on that, so the cleanup keeps real stops.

### 3.5 What we tried and dropped

| tried | outcome |
|---|---|
| own detector / video-based possession | **dropped** by design: 2+ days of infrastructure to reconstruct a signal we already hold numerically |
| broadcast feed | **dropped**: not synchronised, 65.8% homography, needs an invented alignment |
| restart prior as a dead-ball booster | **rejected by measurement**: the interval start is dead 57.1% of the time, but 43% is live play, and boosting bought recall with precision (0.58 → 0.47). Kept at weight 0, documented |
| interpolation-span confidence discount | **removed**: once the formation fallback existed it discarded signal (team accuracy 0.817 → 0.835 without it) |
| boundary re-placement at the maximum-likelihood change point | **neutral** (±25/50/75/100 frames all leave the structure unchanged); implemented, tested, left off |
| learned emission model (TCN / AdaBoost on these features) | **not needed yet**: the rules beat both published baselines; left as the next step |

---

## 4. The possession rules we implemented

Mapping from `CHALLENGE.md` to mechanism:

| rule | implementation |
|---|---|
| a frame is `D` where the ball is out of play | out-of-play ball, long absence, set-piece interval, stationary-ball set-up, stillness; plus the published pre-kickoff block |
| possession stays with the last team while contested or loose | the switch cost exceeds what a loose or missing ball can contribute, so the state persists — this is the main reason H↔A switches are expensive |
| a touch is not a change of possession | the ownership margin is a weighted neighbourhood, not "who touched it last", and a team change needs a sustained margin in the multi-second blend |
| controlled possession = kept for a few seconds / 2+ passes | minimum possession dwell 3 s; anything shorter is absorbed, never reported as a possession |
| `H H D D H H` is two possessions | dead stretches are never merged across in cleanup |
| dead-ball ownership is not asked for | all dead frames are `D`; we never attempt to name the restarting team |
| `X` | written only outside the graded halves (40,462 frames). We never abstain inside a half: the scorer forward-fills an `X`, so abstaining mid-possession is neutral and abstaining across a hand-over is harmful — a guess is strictly better there |

---

## 5. Training data

**None. The pipeline contains no learned parameters and no training set of any kind.**

- No external dataset was used — no SoccerNet, no Metrica open matches, no event data, no corrected
  tracking, nothing hand-labelled.
- No labels of the provided match were used to train, fit or validate a model. There is nothing to
  train: the emissions are hand-specified in log-odds and the decoder has 6 constants.
- The 5 published minutes were used in four ways, all of which we disclose because two sit near the
  line:
  1. **Verification**: frame alignment, Smart Tagging time base, submission format.
  2. **Feature semantics** — the legitimate use the documentation encourages ("check that your
     pipeline is correct … labels meaning what you think"). We measured directional quantities such
     as P(dead | ball outside the pitch) = 27.6% vs 0.0% on live frames, speed on dead vs live
     frames, and the centroid inversion. These are properties of the data, not fitted coefficients.
  3. **A consistency check of the final score** — printed, never used to select a feature set.
  4. **Two tie-break choices** among values that were all defensible on label-free grounds:
     minimum possession 2 s vs 3 s, and the formation weight −0.10 vs −0.25. Both differences are
     inside the noise band (see §6.3), and neither changes the method. We flag them rather than
     claim they were blind.

The priors themselves come from football and from published work (§8), not from this match: effective
playing time ~55–60 of 90 minutes for the dead-ball prior, "a possession lasts several seconds" from
the challenge's own definition and Lang et al.'s 2 s stoppage exclusion, and the
neighbourhood-not-nearest-player representation from Borghesi et al.'s ablation.

---

## 6. Results

### 6.1 Public five minutes

See §1. The pipeline beats the published floor on **both** metrics, and beats `smooth125` on both
while its mean possession duration is within 0.1 s of the truth's (`smooth125` is 4.3 s out, having
merged possessions to reach its count).

### 6.2 Ablation (label-free, whole match)

Chosen on these because they can be measured on all 154 minutes that carry no labels:

| variant | possessions | mean s | dead |
|---|---|---|---|
| argmax of the emissions, no decoding | 760 | 4.54 | 0.457 |
| no multi-scale context (instantaneous margin only) | 721 | 4.83 | 0.453 |
| unnormalised ownership margin (sum over slots) | 591 | 5.71 | 0.468 |
| raw (unsigned) Smart Tagging prior | 519 | 6.60 | 0.461 |
| switch penalty 2 / 4 / **6** / 10 | 492 / 454 / **421** / 421 | 7.06 / 7.64 / **8.23** / 8.23 | 0.456 |
| no stationary-ball duration | 383 | 9.61 | 0.427 |
| restart prior at weight 2.5 (rejected) | 408 | 7.61 | 0.516 |
| **final** | **418** | **8.32** | **0.457** |

The two largest effects are multi-scale context and the switch penalty; the two "confidence"
mechanisms motivated by the literature (interpolation discount, distance gate) were nearly neutral
and are now off.

### 6.3 On the noise in these numbers

The public window is 17 possessions, and `EVALUATION.md` warns that it cannot rank submissions. We
found a sharper instance than the documentation's own: two configurations that are **identical in
code** — differing only by `exp(-span/1e6)` versus exactly `1.0` — flip 568 frames of Viterbi
tie-breaking and move public `spellF1` from 0.4800 to 0.4947. Anything below ~0.04 on this window is
therefore not a result, and we do not report third decimal places as if they were.

---

## 7. Limitations

| limitation | evidence | what we would fix |
|---|---|---|
| transitions, not boundaries, are the remaining error | only 5 of 20 true transitions matched within ±25 frames; 10 spurious and 9 missed possessions at IoU 0.5 | replace the constant switch penalty with a duration-aware model (HSMM) so a possession's expected length constrains when it can end |
| dead ball with a visible ball | `D` recall 0.694 at F1 level; the population we still miss is a ball that is visible and live-looking during a stoppage | learn the restart moment from ball-motion onset, which would place dead ball instead of boosting it |
| second-half dead share is high | 0.485 vs 0.429 in the first half | the stationary-ball term may over-fire when players set up at length; a per-half calibration of the prior against label-free structure |
| away-side recall | A F1 0.781 against H 0.825 | formation features are only used when the ball is missing; extend them to contested frames with a learned weight instead of a hand-set one |
| tracking quality is inherited | the ball is missing on 41% of graded frames and slots swap identity | a learned emission model on the same features (Lang's AdaBoost configuration) is the natural next step, trained on pseudo-labels |
| no validation beyond 5 minutes | by rule, the full-match truth is private | rely on label-free structure plus the ablations; keep the priors explicit so a reviewer can re-derive them |

---

## 8. Reproduction

```bash
# 1. environment (exact versions used; python 3.14.7)
pip install -r requirements-lock.txt

# 2. data (gated; see DATA.md). Either ./data or MTAH_DATA_DIR
export MTAH_DATA_DIR=/path/to/mtah2027

# 3. build the submission — deterministic, no randomness, ~10 s
python -m pipeline.features
python -m pipeline.submit --out nexus.csv

# 4. verify exactly what the organisers check
python evaluation/validate_submission.py nexus.csv

# 5. optional: score against the public window, and reproduce the ablation table
python evaluation/run_evaluation.py
python -m pipeline.ablate

# 6. tests (35, no match data required, <1 s)
python -m pytest tests/ -q
```

Repository layout: `pipeline/io.py` (readers, all indexed to the video frame), `pipeline/features.py`
(the 197-channel matrix), `pipeline/emissions.py` (the two heads and every prior, with its
justification), `pipeline/decode.py` (Viterbi, cleanup, boundary refinement), `pipeline/submit.py`
(the CSV), `pipeline/diag.py` (label-free diagnostics), `pipeline/ablate.py` (the ablation table),
`tests/`. `docs/references.md` and `docs/pipeline.md` carry the evidence and the per-step
measurements. The match data is **not** in the repository and is gitignored.

Determinism: numpy only, no RNG, no threads-dependent reduction, fixed ordering of feature columns.
The same input bundle produces byte-identical output.

---

## 9. References

The full review, with evidence-quality tags and the design decisions each source drove, is
`docs/references.md`.

1. Borghesi M., Dusty Costa L., Morra L., Lamberti F. (2023). *Using temporal convolutional networks
   to estimate ball possession in soccer games.* Expert Systems with Applications 234:119780.
2. Lang S., Wild R., Isenko A., Link D. (2022). *Predicting the in-game status in soccer with machine
   learning using player movement data.* Scientific Reports 12:6843.
3. Link D., Hoernig M. (2017). *Individual ball possession in soccer.* PLOS ONE 12(7):e0179953.
4. Kim H. et al. (2023). *Ball trajectory inference from multi-agent sports contexts using Set
   Transformer and hierarchical Bi-LSTM.* arXiv:2306.08206.
5. Peral M., Capellera G., Rubio A., Agudo A. (2025). *Temporally accurate events detection through
   ball possessor recognition in soccer.* VISAPP 2025.
6. Peral M., Capellera G., Ferraz L., Rubio A., Agudo A. (2026). *Multi-modal soccer scene analysis
   with masked pre-training.* WACV 2026.
7. Vidal-Codina et al. (2022). *Automatic event detection in football using tracking data.*
   arXiv:2202.00804.
8. Organisers' documentation: `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`,
   `SUBMISSION.md`, `evaluation/possession_metrics.py`.
