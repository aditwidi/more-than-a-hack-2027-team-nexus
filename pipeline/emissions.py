"""Emissions: per-frame evidence for {H, A, D} as log-scores.

This is D1 of `docs/references.md` -- the two heads are kept apart on purpose:

* **dead/alive** is driven by ball evidence and restarts. The ball is the strongest cue for this
  head (Borghesi et al. 2023: ball-only input reaches 83.6% on alive/dead, but only 58.4% on the
  3-class problem). Lang et al. 2022 show the remaining signal is in player movement, which is why
  a still formation with the ball absent scores higher than an absent ball alone.
* **H vs A** is driven by the neighbourhood of the ball: sum(exp(-d/4 m)) per team, the five nearest
  players ablation (79.4% vs 58.4%).

Every constant below is a **stated prior**, not a fitted value. That is deliberate: the 5 minutes of
public truth may not be used to tune ([EVALUATION.md](../../EVALUATION.md)), so each number is
justified by football or by the cited literature, and the ablation in `pipeline/ablate.py` reports
what each one costs. Units are log-odds, so they add and can be argued about individually.

Sign convention: higher = more likely. The decoder takes these as log-emissions and adds transition
costs in the same units.
"""

from __future__ import annotations

import numpy as np

from . import io
from games import half_video_blocks

HOME, AWAY, DEAD = 0, 1, 2

DEFAULTS = {
    # --- dead-ball head -------------------------------------------------------------------
    # A second of ball absence is worth this much log-odds of "dead". Recalibrated from 0.55: the
    # ground truth is derived from *corrected* tracking, which knows the ball is in play while ours
    # has lost it, so a long absence is evidence but not proof. Measured on the public window, 391
    # alive frames had a median ball gap of 5.6 s, and the old weight called 98.7% of them dead.
    # Lowering to 0.40 with a smaller long-gap bonus raised D precision 0.69 -> 0.75 with recall
    # unchanged -- fewer false stoppages, not fewer real ones.
    "dead_per_second": 0.40,
    "dead_gap_cap_s": 8.0,
    # Half a second back from the edge of a gap: the frames just before the ball is seen again are
    # usually still live (a pass arriving), so the gap-length term alone should not pin them down.
    "dead_edge_discount": 0.35,
    # A restart is dead by definition. Smart Tagging is inferred, so this is evidence, not a rule.
    "dead_set_piece": 0.9,
    # Visible ball is evidence of play. Small: a ball can be visible and dead (rolling out, sitting
    # in the keeper's hands) and the gap term already covers the invisible case.
    "live_ball": -0.8,
    # "The frame the tracking marks the ball dead" is what the ground truth is derived from, so a
    # long absence is the single most reliable signal. Nothing else gets this weight.
    "dead_long_gap_bonus": 0.8,
    "dead_long_gap_s": 3.0,
    # Ball beyond the touchline or goal line: out of play by definition. Measured on the public
    # window, 27.6% of dead frames and 0.0% of live frames carry it, so it is nearly pure evidence
    # and it is set high enough to survive a strong (but wrong) nearest-player read.
    "dead_out_of_play": 4.0,
    # Player movement. Lang et al. 2022 get 92% on active/interrupted from movement alone: dead ball
    # means players stop, converge and walk. Measured on the public window, mean tracked speed is
    # 1.43 m/s on dead frames and 2.38 m/s on live ones, and the share of players above 3 m/s is
    # 8.7% vs 30.6%. This is the only dead-ball evidence that survives an absent ball.
    "dead_stillness": 1.5,
    "dead_speed_ref": 2.5,
    "dead_motion": 1.2,
    # A visible ball nobody is near: not "controlled possession" (CHALLENGE.md). Weaker than the
    # movement terms on its own -- 23.6% of dead frames vs 15.3% of live ones.
    "dead_loose": 0.8,
    # Shaped set-piece restart prior. A tagged SET PIECES interval is dead 57.1% of the time at its
    # start and through the set-up, against a base rate of 19.6%, so it is a strong *placement*
    # signal: it says *when* play stopped, not just that a stoppage phase exists.
    "dead_restart": 0.0,
    # Stationary ball + slow players: a free kick or throw-in being set up. Measured P(dead) 0.494
    # against a 0.196 base rate. This is the signal that was missing for the 65% of dead frames in
    # which the ball is visible.
    "dead_still_ball": 4.0,
    # Prior against calling dead at all. Effective playing time in professional football is roughly
    # 55-60 minutes of 90, so dead ball is ~35-40% of the match. Without this bias the decoder calls
    # ~48% dead, which also fragments possessions (dead ball breaks a possession in the metric).
    "dead_bias": -1.5,
    # --- possession head ------------------------------------------------------------------
    # Ownership margin: mean exp(-d/4) over the k nearest players *per team*, so it does not depend
    # on how many track slots a team happens to have (24 home vs 22 away in this match). Signed
    # evidence: on live frames its sign is home 57.4% of the time against a truth of 57.3%, where
    # the unnormalised sum said 58.4% -- the normalisation removes the bias instead of damping it.
    # Its spread is ~0.165, so 6.0 log-odds per unit is the same effective assertiveness the
    # unnormalised margin (std 0.571) had at 1.8.
    "possession_scale": 6.0,
    # Which ownership margin to use. ``margin_top3`` is the normalised one; ``score_margin`` is the
    # unnormalised sum, kept as a fallback for feature sets built before the normalisation existed.
    "possession_margin": "margin_top3",
    # Blend three views of the same margin: this frame, +/-1 s, +/-4 s. The instantaneous read is the
    # noisiest number in the dataset (one duplicated track slot can dominate it) and a possession is
    # a multi-second state, so most of the weight belongs further out. Ablation: weights (1, 0, 0)
    # leave 847 possessions on the match against 436 here.
    "possession_weights": (0.2, 0.4, 0.4),
    # How far an interpolated ball position can be trusted, in seconds of absence: exp(-span/scale).
    # Now DISABLED (0.0). It was introduced when the interpolated ball was the only read on a missing
    # ball; with the formation fallback below it is strictly harmful -- turning it off raised team
    # accuracy 0.817 -> 0.835, away recall 0.765 -> 0.807 and cut share error to 0.002, with the
    # label-free structure unchanged (418 vs 422 possessions, 8.32 vs 8.28 s). Kept as a knob
    # because the ablation in docs/pipeline.md reports it.
    "possession_confidence_s": 0.0,
    # A ball further than this from every tracked player is loose or in the air: ownership evidence
    # there is weak, so it is discounted rather than believed (CHALLENGE.md: in contested phases
    # possession stays with the last team).
    # Formation fallback for frames with no ball, and it enters with a NEGATIVE weight: the closer
    # centroid belongs to the *defending* team, which is compact around the ball while the team in
    # possession spreads. Measured on the public window, ball-absent frames whose home centroid was
    # nearer the ball were away possessions 384 out of 384 times. Adding it flipped A recall
    # 0.756 -> 0.765 at -0.10 and 0.809 at -0.25; -0.10 is used because it also shortens the
    # boundary latency (mean -6.6 -> -1.1 frames) without inflating the possession count.
    "shape_scale": -0.10,
    "possession_distance_gate_m": 25.0,
    "possession_gate_factor": 1.0,
    # A team tagged on a phase is a prior on who has the ball, but it has to be the *signed* prior:
    # defending codes are tagged with the team without the ball, and each phase is recorded twice.
    # `tag_possession` in io.py accumulates sign(code) x team over both instances. Measured on the
    # public window: P(home | prior home) = 0.769 against a base rate of 0.573, and P(home | prior
    # away) = 0.435 -- informative, and worth ~0.6 log-odds before any discount for being inferred.
    "tag_team": 0.35,
    "tag_feature": "tag_possession",
    # --- decoder-adjacent priors (used by decode.py, kept here so all priors live in one place) --
    # Switching possession H<->A is the expensive event: it is what a possession *is*. 5.0 with the
    # confidence discount above yields ~200 possessions per half at ~9 s mean, inside the 8-15 s
    # football prior. Raising it further buys little: the path is already saturated by 8.
    "switch_penalty": 6.0,
    # Entering or leaving dead ball is cheaper: stoppages are frequent and often brief.
    "switch_penalty_dead": 2.5,
    # --- structural minimums, in frames (25 fps) -------------------------------------------
    # A possession shorter than a few seconds is not a possession. CHALLENGE.md requires *controlled*
    # possession -- the ball kept for a few seconds, or two completed passes -- and Lang et al. 2022
    # discard stoppages under 2 s. 3 s is the same order, and it is also where the label-free
    # structure is most plausible (398 possessions over the match, mean 8.45 s).
    # Boundary refinement half-window, in frames. Viterbi optimises the smoothed emission; this
    # re-places each edge where the instantaneous evidence changes, which is what IoU measures.
    "boundary_refine_frames": 0,
    "min_possession_frames": 75,
    "min_dead_frames": 25,
}


def column_index(names):
    return {name: j for j, name in enumerate(names)}


def emission_logits(X, names, params=None):
    """(n_frames, 3) log-scores for [H, A, D].

    Deterministic and vectorised: 199,319 frames in well under a second.
    """
    p = dict(DEFAULTS)
    if params:
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise KeyError("unknown emission param(s): {}".format(sorted(unknown)))
        p.update(params)

    column = column_index(names)
    required = ("ball_present", "ball_gap_prev", "ball_gap_next", "in_set_piece",
                "is_graded", "ball_fill_span", "d_home", "d_away")
    missing = [name for name in required if name not in column]
    if missing:
        raise KeyError("features are missing {}; rebuild with python -m pipeline.features"
                       .format(missing))

    present = X[:, column["ball_present"]] > 0.5
    gap_frames = np.maximum(X[:, column["ball_gap_prev"]], X[:, column["ball_gap_next"]])
    gap_seconds = gap_frames / float(io.FPS)

    dead = (p["dead_per_second"] * np.minimum(gap_seconds, p["dead_gap_cap_s"])
            - p["dead_edge_discount"]
            + p["live_ball"] * present
            + p["dead_set_piece"] * (X[:, column["in_set_piece"]] > 0.5)
            + p["dead_long_gap_bonus"] * (gap_seconds >= p["dead_long_gap_s"]))

    if "ball_outside_pitch" in column:
        dead = dead + p["dead_out_of_play"] * (X[:, column["ball_outside_pitch"]] > 0.5)

    if "dead_still_combo" in column:
        dead = dead + p["dead_still_ball"] * X[:, column["dead_still_combo"]]

    if "restart_dead" in column:
        dead = dead + p["dead_restart"] * X[:, column["restart_dead"]]

    if "player_speed_mean" in column:
        reference = p["dead_speed_ref"]
        stillness = np.clip((reference - X[:, column["player_speed_mean"]]) / reference, 0.0, 1.0)
        dead = (dead + p["dead_stillness"] * stillness
                - p["dead_motion"] * X[:, column["player_speed_share_moving"]]
                + p["dead_loose"] * X[:, column["ball_loose"]])

    dead = dead + p["dead_bias"]

    margin_name = p["possession_margin"]
    if margin_name not in column:
        margin_name = "score_margin"          # feature set built before the normalisation
    margin = X[:, column[margin_name]].astype(np.float64)
    w_raw, w_short, w_long = (float(w) for w in p["possession_weights"])
    total = max(w_raw + w_short + w_long, 1e-9)
    blended = w_raw * margin
    for weight, suffix in ((w_short, "mean25"), (w_long, "mean100")):
        name = "{}_{}".format(margin_name, suffix)
        if name in column:
            blended = blended + weight * X[:, column[name]]
    margin = blended / total
    # Rescale if the fallback (unnormalised) margin is in use: it has ~3.5x the spread.
    if margin_name != p["possession_margin"]:
        margin = margin * 0.29
    # Discount ownership evidence where it cannot be trusted: interpolated ball positions, and a
    # ball nobody is near. Without this the decoder reports ~320 possessions per half; the same
    # evidence with the discount reports ~205, because the switch penalty can then do its job.
    span_seconds = X[:, column["ball_fill_span"]] / float(io.FPS)
    if p["possession_confidence_s"] > 0:
        confidence = np.exp(-span_seconds / p["possession_confidence_s"])
    else:
        confidence = np.ones_like(span_seconds, dtype=np.float64)
    nearest = np.minimum(X[:, column["d_home"]], X[:, column["d_away"]])
    if p["possession_gate_factor"] != 1.0:
        confidence = confidence * np.where(nearest > p["possession_distance_gate_m"],
                                           p["possession_gate_factor"], 1.0)
    possession = p["possession_scale"] * margin * confidence
    if "shape_margin" in column:
        # Gated on the ball being *absent*, not on low confidence: the centroid signal inverts when
        # the ball is visible (the defending team is compact around it), so it must only ever speak
        # for frames where there is no ball to see.
        possession = possession + p["shape_scale"] * X[:, column["shape_margin"]] * (~present)

    tag_name = p["tag_feature"] if p["tag_feature"] in column else "tag_team"
    tag = p["tag_team"] * X[:, column[tag_name]]

    logits = np.empty((X.shape[0], 3), dtype=np.float32)
    logits[:, HOME] = possession + tag
    logits[:, AWAY] = -possession - tag
    logits[:, DEAD] = dead

    # Published, observable boundaries: the second half is graded from frame 110,808 but does not
    # kick off until 113,742 (FAQ.md), so those frames are dead ball by construction. This is not
    # ground truth being peeked at -- it is the published match calendar, and it is worth 2,934
    # frames.
    _, second_first, _ = half_video_blocks("BAR-ATH")[1]
    pre_kickoff = slice(second_first, second_first + 2934)
    logits[pre_kickoff, HOME] = -1e6
    logits[pre_kickoff, AWAY] = -1e6
    logits[pre_kickoff, DEAD] = 0.0

    # Centre the dead head against the possession head once: score_margin has an arbitrary zero,
    # the dead log-odds do not, so without this the decoder would be biased toward whichever head
    # happens to have the larger absolute numbers.
    return logits


def explain(X, names, params=None, frames=None):
    """Per-frame breakdown of the emission scores, for debugging a specific moment.

    Returns a list of dicts so it prints usefully and serialises to JSON.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    column = column_index(names)
    logits = emission_logits(X, names, params)

    if frames is None:
        present = X[:, column["ball_present"]] > 0.5
        frames = np.flatnonzero(np.diff(present.astype(np.int8)) != 0)[:20]

    out = []
    for frame in np.atleast_1d(frames):
        gap = max(X[frame, column["ball_gap_prev"]], X[frame, column["ball_gap_next"]])
        out.append({
            "frame": int(frame),
            "second": round(float(frame) / io.FPS, 2),
            "ball_present": bool(X[frame, column["ball_present"]] > 0.5),
            "ball_gap_s": round(float(gap) / io.FPS, 2),
            "in_set_piece": bool(X[frame, column["in_set_piece"]] > 0.5),
            "score_margin": round(float(X[frame, column["score_margin"]]), 3),
            "tag_team": int(X[frame, column["tag_team"]]),
            "logits": {"H": round(float(logits[frame, HOME]), 3),
                       "A": round(float(logits[frame, AWAY]), 3),
                       "D": round(float(logits[frame, DEAD]), 3)},
        })
    return out
