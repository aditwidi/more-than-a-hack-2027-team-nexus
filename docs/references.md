# Literature review: how to improve the possession pipeline

Working notes for **More than a Hack 2027** (frame-level ball possession, Metrica Sports × FC Barcelona).
Compiled 2026-11-11. Purpose: ground every design decision in published evidence, and keep a citable
record for the technical document required by [SUBMISSION.md](../SUBMISSION.md).

**Read the evidence-quality section (§10) before quoting any number from this file.** Some figures
were read from open-access full text; others were summarised by a search provider because the
publisher PDF was access-controlled. Each claim is tagged `[full text]`, `[abstract]`, or
`[summary]`.

---

## 0. TL;DR — eight findings that drive the design

1. **There is exactly one paper on this task**: Borghesi et al. 2023, *{home, away, dead} per frame
   from tracking data*. 86.2% possession accuracy, 89.2% dead/alive. `[abstract]`
2. **Ball-only input is bad at possession (58.4%) and good at dead/alive (83.6%); the five nearest
   players are the reverse (79.4% on the 3-class problem).** So possession and dead-ball are two
   different sub-problems and must not share one decision rule. `[summary]`
3. **Dead/alive can be classified from player positions alone** at 92.0% accuracy / 93.1% F1 using
   AdaBoost on ~1,320 tabular features. The ball is not required. `[summary]`
4. **Multi-scale time shifts are the cheapest feature win in the literature**: position, velocity
   and acceleration of all 22 players at 15 log-spaced lags from 0.4 s to 80 s. `[summary]`
5. **A min-duration stage is standard**, not a hack: a 3.64 s median filter for stoppage extraction,
   with <2 s stoppages excluded from evaluation. Our "merge/denoise possessions" layer is the same
   family of method. `[summary]`
6. **Realistic boundary accuracy is ~1.5–2 s** (1.8 s start, 1.7 s end timing error for stoppages).
   Frame-exact boundaries are not achievable; short possessions are where IoU dies. `[summary]`
7. **Missing-data indicators are features, not noise.** Borghesi feeds explicit missing-data flags.
   With our ball absent on 41% of graded frames, *the pattern of absence* is evidence. `[summary]`
8. **No paper optimises possession-level metrics.** All of them maximise frame accuracy or F1; our
   ranking metric (`spellF1`, one-to-one possession matching at IoU 0.3–0.7) is unaddressed by the
   literature. The structured decoding layer is therefore where we can beat the published methods,
   and the challenge's own baseline sweep already shows a 5× swing from smoothing alone.

---

## 1. The task-defining paper

### Borghesi, Dusty Costa, Morra, Lamberti (2023) — *Using Temporal Convolutional Networks to Estimate Ball Possession in Soccer Games*
*Expert Systems with Applications* 234:119780 · DOI [10.1016/j.eswa.2023.119780](https://doi.org/10.1016/j.eswa.2023.119780)
· accepted-version PDF: [iris.polito.it](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)

The only paper we found whose output space matches this challenge exactly: a per-frame label of
*home possession / away possession / dead ball* from optical tracking.

**Inputs.** Sequences of tracked objects; features per object: coordinates, velocity, **team**,
**role**, player identity, and **missing-data indicators**. Objects are embedded individually and
combined with permutation-invariant aggregation (so the model does not depend on track ordering —
directly relevant to our 46 anonymous, swapping track slots).

**Architecture.** Temporal Convolutional Networks; the best variant is **two branches with
self-attention**. Causal or acausal context both supported — future frames help accuracy but are
unsuitable for strict real-time. *We are offline, so acausal context is free for us.*

**Results.** 86.2% possession-estimation accuracy (+7.3 pp over reimplemented rule-based
state-of-the-art) and 89.2% dead/alive accuracy (+33.2 pp). Mean overall accuracy across variants
82.93% ± 1.71%. Inference ~16 fps on a consumer GPU (comfortably faster than 25 fps video, i.e.
real-time capable).

**Ablations — the most useful part for us.** `[summary]`

| input | 3-class game state | active/inactive |
|---|---|---|
| ball only | **58.35%** | 83.62% |
| five players closest to the ball | **79.41%** | — |
| full model | 86.2% (possession) | 88.25–89.2% |

Interpretation we adopt: **the ball is the dead-ball detector; the neighbourhood of players is the
possession detector.** A single nearest-player rule (our published baseline) throws away most of
the available signal on *both* heads.

**Caveats stated by the authors.** Dataset is a limited selection of high-quality games with
manually labelled states (~40% dead, roughly equal home/away); train/val/test split by match,
stadium and teams; rule-based competitors were reimplemented with hyperparameters retained from
prior work, so the +7.3/+33.2 pp margins are not a clean comparison.

---

## 2. Dead-ball / "in-game status" detection

### Lang, Wild, Isenko, Link (2022) — *Predicting the in-game status in soccer with machine learning using player movement data*
*Scientific Reports* 12:6843 · [nature.com/articles/s41598-022-19948-1](https://www.nature.com/articles/s41598-022-19948-1)

Binary **active vs interrupted** from 25 fps tracking of all 22 players — **no ball data**.

**Features.** Player position, velocity and acceleration, augmented with **7 past and 7 future time
shifts** spanning ~0.4 s to 80 s → **1,320 features** (≈ 22 players × 4 channels × 15 lags).
Multi-scale context, acausal by design.

**Model.** AdaBoost (classical, CPU-trainable in minutes) beat the other classical models tested.

**Results.** **92.0% frame accuracy, 93.1% frame F1** on 47 test matches (102 Bundesliga matches in
total). For stoppages longer than 2 s: **precision 81.1%, recall 78.9%**, ~0.80 mean F1, with
**start/end timing errors of 1.8 s / 1.7 s**. A **3.64 s median filter** is used to extract stoppage
intervals, and stoppages shorter than 2 s are excluded.

**Why this matters for us.**

- Confirms `D` is detectable from *player behaviour* (players stop, converge, walk back into
  position), which is exactly what we need on the 41% of frames with no ball.
- Confirms a **tabular, gradient-boosted model on engineered multi-scale features** is competitive
  with deep sequence models for this head — a good fit for a 4 GB GTX 1650 Ti and a 3–4 day budget.
- Sets a realistic expectation for **boundary accuracy ~1.5–2 s**. Consequence for `spellF1`: a
  ~1.5 s error on each edge of a 14 s possession gives IoU ≈ 0.80; on a 4 s possession it is fatal.
  **Protect the existence of short possessions; do not chase frame-exact edges.**
- The median filter *is* a structured-decoding step. Published practice agrees with the challenge's
  argument that smoothing/segmentation is worth more than per-frame cleverness.

**Caveats.** Bundesliga only; the authors flag limited generalisation. The method uses future
frames, so it is not deployable in strict real time (irrelevant to us).

---

## 3. When the ball is missing: reconstructing it from players

### Kim et al. (2023) — *Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM* ("Ball Radar")
arXiv [2306.08206](https://arxiv.org/abs/2306.08206) · code: [github.com/hyunsungkim-ds/ballradar](https://github.com/hyunsungkim-ds/ballradar)

Explicitly motivated by our exact problem: a tiny ball on a wide pitch, lost to occlusion.

**Approach.** Set Transformer over the multi-agent configuration (permutation-invariant, no
jersey-order dependence) + a **hierarchical Bi-LSTM** that **predicts ball possession as an
intermediate task** and then regresses the ball trajectory, with a physical "reality" loss and
post-processing. Applications listed include **missing-trajectory imputation**.

**Why it matters.** It is evidence that player context can stand in for an absent ball, and that
possession is a useful *intermediate representation* rather than only an output. We are not going
to train this in the time available, but it justifies the cheap analogue: predict/impute the ball
from the surrounding players before making a possession decision.

**Caveats.** No quantitative results were retrievable from the abstract page `[abstract]`; training
cost is far beyond the budget; transfer to Metrica's automatic tracking is unproven.

---

## 4. Ball possessor recognition from video (the road not taken)

### Peral, Capellera, Rubio, Agudo et al. (2025, 2026)
- *Temporally Accurate Events Detection Through Ball Possessor Recognition in Soccer*, VISAPP 2025 ·
  [scitepress PDF](https://www.scitepress.org/Papers/2025/133177/133177.pdf)
- *Multi-Modal Soccer Scene Analysis with Masked Pre-Training*, WACV 2026 ·
  [CVF PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.pdf)

Video-based pipeline: build player video tubes, estimate possession, identify the possessor,
extract passes/receptions. ResNet50/TSM features, temporal modelling, rule-based event extraction.
Evaluated on **36 La Liga and Serie A matches** with frame-level annotations; event timing accurate
to **~0.6 s**. The WACV 2026 paper addresses **tactical camera footage** (our feed) with three heads:
ball trajectory inference, **ball state classification**, and ball possessor identification.

**Why we are not following it.** It needs pixel-level work (video decode, detection, tracking,
tube association) that the rules permit but our calendar does not: ~2+ days of infrastructure
before the first measurable possession label. It is also the *harder* route to a signal we already
hold numerically. Keep it as a documented "considered and dropped" item in the technical doc, and
as a reference point that **~0.6 s event timing is achievable with video-level supervision** —
i.e. our ~1.5 s tracking-only boundary error is respectable, not a failure.

---

## 5. Deterministic rule-based possession (the baseline family)

### Vidal-Codina et al. (2022) — *Automatic event detection in football using tracking data*
arXiv [2202.00804](https://arxiv.org/abs/2202.00804)

Two-stage deterministic decision-tree method: (1) a **possession step** deciding which player has
the ball per frame, (2) event rules on top (passes, receptions, etc.). This is the family our
published baseline belongs to — nearest tracked player to the ball, then a majority vote.

**Relevance.** It is the "rule-based state of the art" that Borghesi's +7.3 pp and Lang's margins are
measured against, which tells us how much headroom there is in upgrading the rules themselves
(neighbourhood voting, team shape, holder continuity, multi-scale context) before any learning.

### Link & Hoernig (2017) — *Individual ball possession in soccer*
*PLOS ONE* 12(7):e0179953 · [journals.plos.org](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0179953)

Defines a taxonomy of possession by **start point, end point and type of ball control**: Individual
Ball Control (IBC), Individual Ball Action (IBA), Team Ball Possession (TBP), Team Ball Control
(TBC), Team Playmaking (TPM), with an ML approach to estimate durations.

**Relevance.** This is the formal vocabulary for the ambiguity in CHALLENGE.md ("a touch is not a
change of possession; controlled possession means the player keeps the ball / the team completes
2+ passes"). Adopt its distinctions explicitly in our possession rules rather than inventing
thresholds: distinguish *control* (retains the ball) from *action* (one touch) from *team
possession* (who the ball belongs to while live).

---

## 6. Ball-free possession inference (adjacent, unverified)

### *PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories*
arXiv [2602.12080v2](https://arxiv.org/html/2602.12080v2)

Proposes inferring **possession paths** from player trajectories alone via a CRF — i.e. possession
without any ball track. Directly relevant to the 41% missing-ball frames.

**Status: unverified.** The HTML body could not be extracted (no text matches on retrieval), so
nothing in this note depends on it. Treat as a lead for a future iteration, not as evidence.

---

## 7. What the literature does *not* solve for this competition

| gap | detail | consequence |
|---|---|---|
| **Possession-level metrics** | Every paper reports frame accuracy/F1. None reports possession segmentation quality, count error, or one-to-one matching. | A frame-level SOTA system can rank badly here. The score is made in the decoding layer, not the classifier. |
| **No in-domain training labels** | Those papers had vendor- or manually-labelled data from dozens of matches (102 Bundesliga, 36 La Liga/Serie A, "selected games"). We have 7,500 frames we may not train or tune on. | Supervised emission models must be trained on *other games* (allowed) or on pseudo-labels (allowed), both of which carry domain-mismatch or imitation risk. |
| **Their `D` is easier than ours** | Vendor-labelled stoppages, ~40% dead, clean definitions. | Our `D` is derived from corrected tracking, includes the 2,934 all-`D` frames before the H2 kickoff, and our ball is absent 41% of the time. Their dead-ball accuracy is not transferable as-is. |
| **Ball quality** | Their optical tracking is described as high quality; ours is explicitly automatic, uncorrected, 46 swapping anonymous slots, ball missing 41% of graded frames. | All published accuracies are optimistic for us. Our floor (spellF1 0.2433) is the honest reference, not 86%. |

---

## 8. Design decisions derived from the evidence

Ordered by expected gain per day of work. `E` = expected effect on the ranking metric.

### D1 — Split the decision into two heads
Separate **dead/alive** from **H vs A**, with independent evidence, then combine in the decoder.
*Evidence*: Borghesi ablation (ball 58.4% vs top-5 players 79.4%; ball 83.6% on alive/dead); Lang
92% with no ball at all.
*Why it fixes us*: the current floor decides `D` from ball absence (recall 0.21) and H/A from one
nearest player (team accuracy 0.835). Each head is using the other head's weak signal.
`E`: large — this is where `D` recall and `team_acc` are both recovered.

### D2 — Ownership evidence = weighted k-nearest neighbourhood, not nearest player
Distances in metres (105.3 × 68.0), per-team weights over the k nearest, plus team-shape terms
(count of each team's players within a radius of the ball, ball-to-team-centroid ratio) and
**holder continuity** (decaying prior for the last confirmed controller).
*Evidence*: Borghesi top-5-players ablation; "team/role features are most important for possession";
Link & Hoernig's control-vs-action distinction supplies the semantic ("a touch is not a change").
`E`: moderate on frames; large indirectly because clean evidence reduces spurious possessions.

### D3 — `D` by positive evidence, with ball absence as one feature among several
Evidence sources: length and location of ball-track gaps; Smart Tagging `SET PIECES` (throw-in,
corner, goal kick, free kick — every restart ends a dead stretch); kickoff anchors; half
boundaries; formation signature of stoppage (players converging, low aggregate speed) which is what
Lang actually learns.
*Evidence*: Lang (stoppages from player movement, 92%); Borghesi (ball is the strongest dead/alive
cue); local diagnosis (D recall 0.21).
`E`: large — `D` boundaries are what split possessions; a missed `D` merges two possessions into one
wrong one, a phantom `D` splits one possession into two.

### D4 — Multi-scale acausal features
Position, velocity, acceleration (and ball position/visibility flags) at log-spaced lags ~0.4 s to
80 s, centred (future included). Add explicit **NaN/missing indicators** per slot and for the ball.
*Evidence*: Lang's 1,320 features → 92% with AdaBoost; Borghesi feeds missing-data indicators;
acausality declared free for an offline submission.
`E`: moderate directly, large as the substrate for D7.

### D5 — Structured decoding with duration constraints (the differentiator)
Hidden semi-Markov / Viterbi over {H, A, D} with: switch penalty encoding "a possession lasts
~10–20 s"; **min dwell ~1–2 s** per state; **min possession length ~2 s**; merge same-team runs
separated by short `D`; drop sub-minimum possessions.
*Evidence*: Lang's 3.64 s median filter and <2 s stoppage exclusion; the challenge's own baseline
sweep (spellF1 0.0486 → 0.2433 from smoothing alone; `[EVALUATION.md](../EVALUATION.md)`).
*Local measurement*: the floor produces 19 spurious possessions and 29 transitions against 20 real
ones on the public window; removing half the spurious possessions lifts F1@IoU0.5 from 0.326 to
~0.42, removing nearly all of them to ~0.58 (upper bound, ignores the recall cost).
`E`: largest single effect available to us.

### D6 — Do not tune on the 5 public minutes
*Evidence*: `[EVALUATION.md](../EVALUATION.md)` — the window picks the wrong smoothing variant even
for a one-parameter baseline; the possession count/mean duration published for that window is
precisely the quantity one would be tempted to fit.
*Policy*: switch penalty and min-dwell come from football priors (10–20 s possessions) stated as
priors in the technical doc, plus **label-free** full-match sanity checks (does `D` align with
set-piece tags? is the predicted possession count plausible per half? is home share in a plausible
band?), plus `--bootstrap 1000` before believing any delta.

### D7 — Learned emission model: stretch goal, not a commitment
If D1–D5 plateau, fit **AdaBoost / LightGBM on D4 features** (CPU, minutes) — the configuration that
won in Lang — rather than a TCN. Rationale: Lang's tabular model beat sequence models on the
dead/alive head, and our label budget is the binding constraint, not model capacity.
Training labels must come from **other games** (Metrica's open 3-match dataset, same vendor and
format) or from algorithmic pseudo-labels; never from this match.

### D8 — Explicitly documented non-goals
Video-based possessor recognition (Peral), end-to-end Ball Radar, and the broadcast feed. Each is
evidence-backed but costs 2+ days of infrastructure; mention them as *considered and dropped*, with
the reason, in the technical document.

---

## 9. Mapping to the required technical document

[SUBMISSION.md](../SUBMISSION.md) asks for six things; this file supplies the citations for four:

| required section | material here |
|---|---|
| approach and why | §8 D1–D5; §1 ablations justify the two-head design |
| assets used / dropped | §4 and D8 (video, broadcast dropped with reasons) |
| training data | §7 (none from this match); D7 names the only permitted external source |
| possession rules implemented | §5 (Link & Hoernig taxonomy), D2/D3 definitions, D5 duration parameters |
| limitations | §7 table; §2 boundary-timing ceiling; §10 evidence quality |
| reproduction | not covered here — see the implementation repo |

---

## 10. Evidence quality

| claim | status | note |
|---|---|---|
| Borghesi: 86.2% possession / 89.2% dead-alive, +7.3 / +33.2 pp | `[abstract]` | publisher and institutional PDFs returned HTTP 403; figures from the DOI abstract page and search-provider summary of the accepted manuscript |
| Borghesi ablation: ball-only 58.35%, top-5 players 79.41%, ball-only alive/dead 83.62% vs 88.25% | `[summary]` | read from a search-provider synthesis of the accepted version; **re-read the PDF before quoting in the submission** |
| Borghesi inputs include team, role, player ID, missing-data indicators; permutation-invariant aggregation; 16 fps | `[summary]` | as above |
| Lang: 92.0% acc / 93.1% F1; 1,320 features; 15 lags 0.4–80 s; 3.64 s median filter; <2 s excluded; 1.8 s / 1.7 s timing error; 102 matches, 47 test | `[summary]` | synthesised from nature.com article + table 1; the 1,320 = 22 × 4 × 15 arithmetic is **our inference** from the stated count, not a quoted formula |
| Kim et al.: Set Transformer + hierarchical Bi-LSTM, intermediate possession prediction, reality loss | `[abstract]` | no metrics retrieved |
| Peral: 36 matches, ~0.6 s event timing, ResNet50/TSM | `[summary]` | two papers conflated in retrieval; attribute carefully (VISAPP 2025 = possessor + events; WACV 2026 = ball trajectory/state/possessor from tactical footage) |
| PathCRF | **unverified** | body not extractable; do not cite |
| Local numbers: D recall 0.21, 19 spurious possessions, 29 vs 20 transitions, spellF1 by threshold | **verified locally** | `evaluation/reports/baseline_smooth50.json`, `baseline_smooth50_transitions.csv` |

**Action before the deadline:** retrieve the Borghesi accepted PDF (institutional access, or request
from the authors) and the Lang feature appendix, and replace `[summary]` rows with verified text.

---

## 11. Sources

- Borghesi M., Dusty Costa L., Morra L., Lamberti F. (2023). *Using Temporal Convolutional Networks
  to Estimate Ball Possession in Soccer Games.* Expert Systems with Applications 234:119780.
  DOI [10.1016/j.eswa.2023.119780](https://doi.org/10.1016/j.eswa.2023.119780)
- Lang S., Wild R., Isenko A., Link D. (2022). *Predicting the in-game status in soccer with machine
  learning using player movement data.* Scientific Reports 12:6843.
  [doi:10.1038/s41598-022-19948-1](https://doi.org/10.1038/s41598-022-19948-1)
- Kim H. et al. (2023). *Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set
  Transformer and Hierarchical Bi-LSTM.* [arXiv:2306.08206](https://arxiv.org/abs/2306.08206)
- Peral M., Capellera G., Rubio A., Agudo A. et al. (2025). *Temporally Accurate Events Detection
  Through Ball Possessor Recognition in Soccer.* VISAPP 2025.
  [PDF](https://www.scitepress.org/Papers/2025/133177/133177.pdf)
- Peral M., Capellera G., Ferraz L., Rubio A., Agudo A. (2026). *Multi-Modal Soccer Scene Analysis
  with Masked Pre-Training.* WACV 2026. [PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.pdf)
- Vidal-Codina et al. (2022). *Automatic event detection in football using tracking data.*
  [arXiv:2202.00804](https://arxiv.org/abs/2202.00804)
- Link D., Hoernig M. (2017). *Individual ball possession in soccer.* PLOS ONE 12(7):e0179953.
  [doi:10.1371/journal.pone.0179953](https://doi.org/10.1371/journal.pone.0179953)
- (unverified) *PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player
  Trajectories.* [arXiv:2602.12080v2](https://arxiv.org/html/2602.12080v2)

Local, authoritative for scoring: [CHALLENGE.md](../CHALLENGE.md), [DATA.md](../DATA.md),
[EVALUATION.md](../EVALUATION.md), `evaluation/possession_metrics.py`.

---

## 12. BibTeX for the technical document

```bibtex
@article{borghesi2023tcn,
  title   = {Using temporal convolutional networks to estimate ball possession in soccer games},
  author  = {Borghesi, Matteo and Dusty Costa, Lorenzo and Morra, Lia and Lamberti, Fabrizio},
  journal = {Expert Systems with Applications},
  volume  = {234},
  pages   = {119780},
  year    = {2023},
  doi     = {10.1016/j.eswa.2023.119780}
}

@article{lang2022ingame,
  title   = {Predicting the in-game status in soccer with machine learning using player movement data},
  author  = {Lang, Steffen and Wild, Raphael and Isenko, Alexander and Link, Daniel},
  journal = {Scientific Reports},
  volume  = {12},
  pages   = {6843},
  year    = {2022},
  doi     = {10.1038/s41598-022-19948-1}
}

@misc{kim2023ballradar,
  title         = {Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM},
  author        = {Kim, Hyunsung and others},
  year          = {2023},
  eprint        = {2306.08206},
  archivePrefix = {arXiv}
}

@inproceedings{peral2025possessor,
  title     = {Temporally Accurate Events Detection Through Ball Possessor Recognition in Soccer},
  author    = {Peral, Marc and Capellera, Guillem and Rubio, Antonio and Agudo, Antonio},
  booktitle = {Proceedings of VISAPP},
  year      = {2025}
}

@misc{peral2026multimodal,
  title     = {Multi-Modal Soccer Scene Analysis with Masked Pre-Training},
  author    = {Peral, Marc and Capellera, Guillem and Ferraz, Luis and Rubio, Antonio and Agudo, Antonio},
  booktitle = {WACV},
  year      = {2026}
}

@misc{vidaldina2022events,
  title         = {Automatic event detection in football using tracking data},
  year          = {2022},
  eprint        = {2202.00804},
  archivePrefix = {arXiv}
}

@article{link2017possession,
  title   = {Individual ball possession in soccer},
  author  = {Link, Daniel and Hoernig, Martin},
  journal = {PLOS ONE},
  volume  = {12},
  number  = {7},
  pages   = {e0179953},
  year    = {2017},
  doi     = {10.1371/journal.pone.0179953}
}
```

---

## 13. Open decisions (unresolved at time of writing)

1. **Emission strategy**: rules-only (deterministic, no data dependency) vs a trained tabular model
   (AdaBoost/LightGBM on D4 features). Evidence favours learned emissions *if* labels exist; the
   rule-based path is the safe 3–4 day plan with learning as a stretch.
2. **External label source** for D7, if taken: Metrica's open 3-match dataset (same vendor, same EPTS
   format, event CSVs from which possession can be derived algorithmically) — allowed and low-risk
   in format, but only 3 matches and a different league/competition context.
3. **`D` edge refinement**: whether to spend effort narrowing the ~1.7–1.8 s boundary error
   (belief: expensive, low payoff) or to spend it on possession-level structure (belief: cheap, high
   payoff).
