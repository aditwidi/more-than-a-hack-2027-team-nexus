"""Build one per-frame feature matrix from the tracking, tags and homography.

Design notes live in `docs/references.md`; this file is the implementation of D2 (ownership
evidence), D3 (dead-ball evidence) and D4 (multi-scale acausal context). It makes **no decisions**
-- it emits evidence, and the decoder (Step 2) turns evidence into a label sequence.

Why these features, briefly, with the evidence they come from:

* **k-nearest team evidence, not nearest player** -- Borghesi et al. 2023 ablation: ball-only input
  is 58.4% on the 3-class task while the five players closest to the ball reach 79.4%.
* **Ball gap lengths as features** -- the ball is absent on 41% of graded frames (DATA.md). The
  *pattern* of absence is evidence for `D`, and missing-data indicators are explicit model inputs
  in Borghesi et al.
* **Team shape** -- dead ball is detectable from player movement alone at 92% accuracy (Lang et al.
  2022), which is the only signal available when the ball is invisible.
* **Multi-scale lags, acausal** -- Lang et al. use position/velocity/acceleration at 15 log-spaced
  lags from 0.4 s to 80 s. We are offline, so future context is free (Borghesi et al. note it helps).

Output contract: an ``X`` array of shape (199319, n_features) in float32, plus the list of feature
names. Every row is a video frame, so row i of a submission is row i of X.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from . import io
from games import half_video_blocks

# ---------------------------------------------------------------------------------------------
# Tunables. Metres and seconds, so they can be argued about in the technical document.
# ---------------------------------------------------------------------------------------------
TAU_M = 4.0                 # soft-nearest decay: exp(-d/tau). ~4 m is "within playing distance"
RADII_M = (5.0, 10.0)       # "in the neighbourhood of the ball" radii
TOP_K = 3                   # players per team for the normalised ownership score
DISTANCE_CAP_M = 60.0       # cap for nearest-player distances, so a missing team cannot dominate
GAP_CAP_FRAMES = 250        # 10 s: longer absences are all "very long"
LAGS_FRAMES = (10, 25, 50, 100, 200, 400, 800, 1600)   # 0.4, 1, 2, 4, 8, 16, 32, 64 s
ROLLING_HALVES = (25, 100, 400)                          # ±1 s, ±4 s, ±16 s

# The channels that get lag/rolling treatment: the ones a decoder would actually want history for.
LAGGED_CHANNELS = ("margin_top3", "d_ratio", "ball_present", "n_diff_r5",
                   "ball_x", "ball_y", "home_player_share")

ARTIFACTS = os.path.join(io.ROOT, "artifacts")


# ---------------------------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------------------------
def ball_presence(ball):
    """Visibility flag and the length of the surrounding absence, in frames.

    0/0 means "seen on this frame". A frame deep inside an absence has both large; a frame at the
    edge of one has one large and one small, which is how an edge can be located at all.
    """
    present = ~np.isnan(ball[:, 0])
    index = np.arange(len(ball))

    previous = np.maximum.accumulate(np.where(present, index, -1))
    following = np.minimum.accumulate(np.where(present, index, len(ball))[::-1])[::-1]

    prev_gap = np.where(present, 0, index - np.maximum(previous, 0))
    next_gap = np.where(present, 0, np.minimum(following, len(ball) - 1) - index)
    return present, np.minimum(prev_gap, GAP_CAP_FRAMES), np.minimum(next_gap, GAP_CAP_FRAMES)


def fill_ball(ball, present):
    """Interpolate the ball across absences, and say how far the guess reaches.

    Linear between the last and next observation; held at the nearest observation when the
    absence touches the start or end of the source. Every filled value carries ``fill_span``
    so the decoder can discount it -- a 2-frame interpolation is nearly free, a 60-frame one is
    a guess and should not on its own overturn a possession.
    """
    index = np.arange(len(ball), dtype=np.float64)
    known = present.astype(np.float64)

    previous = np.maximum.accumulate(np.where(present, index, -1.0))
    following = np.minimum.accumulate(np.where(present, index, index[-1] + 1.0)[::-1])[::-1]
    following = np.where(np.isposinf(following), index[-1], following)

    prev_index = np.clip(previous, 0, len(ball) - 1).astype(np.int64)
    next_index = np.clip(following, 0, len(ball) - 1).astype(np.int64)

    span = np.maximum(next_index - prev_index, 1)
    weight = np.clip((index - prev_index) / span, 0.0, 1.0)[:, None]

    filled = np.where(np.isnan(ball[:, :2][prev_index]),
                      ball[:, :2][next_index],
                      ball[:, :2][prev_index] * (1.0 - weight) + ball[:, :2][next_index] * weight)
    filled = np.where(present[:, None], ball[:, :2], filled)
    # A source that never sees the ball at all (or only at one end) falls back to the centre spot.
    filled = np.where(np.isnan(filled), 0.5, filled).astype(np.float32)

    span_out = np.where(present, 0.0, np.minimum(span, GAP_CAP_FRAMES)).astype(np.float32)
    if not present.any():
        # No observation anywhere: the centre spot is a placeholder, not an interpolation, so it
        # must not read as high-confidence evidence anywhere.
        span_out[:] = 0.0
    return filled, span_out, known


def ball_motion(filled, present):
    """Ball velocity in metres/sec from the filled trajectory, zeroed across long gaps."""
    step = np.diff(filled, axis=0, prepend=filled[:1])
    metres = np.stack([step[:, 0] * io.PITCH[0], step[:, 1] * io.PITCH[1]], axis=1)
    speed = np.linalg.norm(metres, axis=1)
    # A jump across a long absence is not motion, it is a re-acquisition: zero it out.
    speed = np.where(present, speed, 0.0)
    return metres[:, 0].astype(np.float32), metres[:, 1].astype(np.float32), speed.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------------------------
def player_metres(slots):
    """Slots in metres, (n_frames, n_slots, 2), NaN preserved."""
    metres = np.empty(slots.shape[:2] + (2,), dtype=np.float32)
    metres[..., 0] = slots[..., 0] * io.PITCH[0]
    metres[..., 1] = slots[..., 1] * io.PITCH[1]
    return metres


def _nanmin_rows(values, mask):
    """Row-wise minimum over the columns selected by ``mask``; +inf where every entry is NaN."""
    selected = np.where(mask[None, :], values, np.inf)
    selected = np.where(np.isnan(selected), np.inf, selected)
    return selected.min(axis=1)


def ownership_evidence(metres, slot_teams, ball_metres):
    """Per-team evidence that each team is in control of the ball, frame by frame.

    Distances are in metres. ``score_*`` is a soft count of that team's tracked players near the
    ball, sum(exp(-d/tau)): one player at 2 m counts more than three at 12 m, and it degrades
    gracefully when the tracker duplicates somebody (Borghesi et al. feed exactly this kind of
    object-level evidence rather than a single nearest-player answer).
    """
    dx = metres[..., 0] - ball_metres[:, :1]
    dy = metres[..., 1] - ball_metres[:, 1:2]
    distance = np.sqrt(dx * dx + dy * dy)

    home = slot_teams == 0
    away = ~home

    d_home = _nanmin_rows(distance, home)
    d_away = _nanmin_rows(distance, away)

    weight = np.exp(-np.where(np.isnan(distance), np.inf, distance) / TAU_M)
    weight = np.where(np.isnan(weight), 0.0, weight)
    score_home = weight[:, home].sum(axis=1)
    score_away = weight[:, away].sum(axis=1)

    out = {
        "d_home": np.minimum(d_home, DISTANCE_CAP_M),
        "d_away": np.minimum(d_away, DISTANCE_CAP_M),
        "score_home": score_home,
        "score_away": score_away,
    }
    out["d_ratio"] = out["d_away"] - out["d_home"]          # >0 means home is nearer
    out["score_margin"] = score_home - score_away
    out["score_total"] = score_home + score_away

    for radius in RADII_M:
        out["n_home_r{}".format(int(radius))] = (distance[:, home] < radius).sum(axis=1)
        out["n_away_r{}".format(int(radius))] = (distance[:, away] < radius).sum(axis=1)
        out["n_diff_r{}".format(int(radius))] = (out["n_home_r{}".format(int(radius))]
                                                - out["n_away_r{}".format(int(radius))])

    # Normalised ownership: mean weight of the k nearest players *per team*.
    #
    # The sum above is biased toward whichever team has more track slots -- this match has 24 home
    # slots and 22 away ones (DATA.md), and the bias showed up as a predicted alive home share of
    # 0.71 against a truth of 0.57. A mean over the k nearest players is invariant to how many slots
    # a team happens to be tracked with, and it matches the "five players closest to the ball"
    # representation that Borghesi et al. found strongest for this head.
    for name, mask in (("home", home), ("away", away)):
        block = distance[:, mask]
        k = min(TOP_K, block.shape[1])
        nearest = np.partition(block, k - 1, axis=1)[:, :k]
        weights = np.exp(-np.where(np.isnan(nearest), np.inf, nearest) / TAU_M)
        weights = np.where(np.isnan(weights), 0.0, weights)
        valid = np.sum(~np.isnan(nearest), axis=1)
        out["score3_{}".format(name)] = np.where(valid > 0, weights.sum(axis=1) / np.maximum(valid, 1), 0.0)

    out["margin_top3"] = out["score3_home"] - out["score3_away"]
    return out


def team_shape(metres, slot_teams, ball_metres):
    """Centroid, spread and ball-to-centroid distance per team.

    Possession is not only proximity: the team whose shape sits between the ball and its own goal,
    and whose players are close together around the ball, is the team playing. These are the
    "team/role" features Borghesi et al. found most useful for the possession head.
    """
    out = {}
    for name, mask in (("home", slot_teams == 0), ("away", slot_teams != 0)):
        block = metres[:, mask]
        count = np.sum(~np.isnan(block[..., 0]), axis=1)

        with warnings.catch_warnings():
            # Before kickoff no slot is tracked at all, so a team's mean is of an empty slice.
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore"):
                centroid = np.nanmean(block, axis=1)
                dx = block[..., 0] - centroid[:, :1]
                dy = block[..., 1] - centroid[:, 1:2]
                spread = np.sqrt(np.nanmean(dx * dx + dy * dy, axis=1))

        centroid = np.where(np.isnan(centroid), 0.5 * np.array(io.PITCH), centroid)
        out["{}_cx".format(name)] = centroid[:, 0]
        out["{}_cy".format(name)] = centroid[:, 1]
        out["{}_spread".format(name)] = np.where(np.isnan(spread), 0.0, spread)
        out["n_{}_tracks".format(name)] = count

        grad_x = centroid[:, 0] - ball_metres[:, 0]
        grad_y = centroid[:, 1] - ball_metres[:, 1]
        out["d_centroid_{}".format(name)] = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    # Formation as a fallback ownership signal, in metres: positive = home centroid closer to the
    # ball. Measured on alive frames where the ball is MISSING, this is nearly pure evidence
    # (384/384 frames with an away-closer centroid were away possessions), and on frames where the
    # ball is visible it *inverts* -- the defending team is compact around the ball while the
    # attacking team spreads -- so the emission gates it to unreliable-ball frames only.
    out["shape_margin"] = (out["d_centroid_away"] - out["d_centroid_home"]).astype(np.float32)

    total = out["n_home_tracks"] + out["n_away_tracks"]
    out["home_player_share"] = np.where(total > 0, out["n_home_tracks"] / np.maximum(total, 1), 0.5)
    return out


# ---------------------------------------------------------------------------------------------
# Multi-scale context
# ---------------------------------------------------------------------------------------------
def shift(values, offset):
    """Value at t-offset, edge-clamped (never wrapped: a wrap would invent future at frame 0)."""
    if offset == 0:
        return values
    out = np.empty_like(values)
    if offset > 0:
        out[offset:] = values[:-offset]
        out[:offset] = values[0]
    else:
        out[:offset] = values[-offset:]
        out[offset:] = values[-1]
    return out


def rolling_mean(values, half_window):
    """Centred mean over +/-half_window, ignoring NaN."""
    values = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(values)
    total = np.cumsum(np.where(valid, values, 0.0), dtype=np.float64)
    count = np.cumsum(valid, dtype=np.int64)
    total = np.concatenate(([0.0], total))
    count = np.concatenate(([0], count))

    index = np.arange(len(values))
    low = np.clip(index - half_window, 0, len(values))
    high = np.clip(index + half_window + 1, 0, len(values))

    summed = total[high] - total[low]
    counted = count[high] - count[low]
    return np.where(counted > 0, summed / np.maximum(counted, 1), np.nan)


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------
def build(data_dir=None, verbose=True):
    """Assemble the full feature matrix. Returns (X float32, names list[str])."""
    assets = io.load_all(data_dir=data_dir)
    n_frames = io.N_FRAMES
    slots, ball, slot_teams = assets["slots"], assets["ball"], assets["slot_teams"]

    metres = player_metres(slots)
    present, prev_gap, next_gap = ball_presence(ball)
    filled, fill_span, _ = fill_ball(ball, present)
    ball_metres = np.stack([filled[:, 0] * io.PITCH[0], filled[:, 1] * io.PITCH[1]], axis=1)
    ball_vx, ball_vy, ball_speed = ball_motion(filled, present)

    columns = {}

    # --- ball ------------------------------------------------------------------------------
    columns["ball_present"] = present.astype(np.float32)
    columns["ball_gap_prev"] = prev_gap.astype(np.float32)
    columns["ball_gap_next"] = next_gap.astype(np.float32)
    columns["ball_fill_span"] = fill_span
    columns["ball_x"], columns["ball_y"] = filled[:, 0], filled[:, 1]
    columns["ball_speed_atd"] = np.nan_to_num(ball[:, 2], nan=0.0)
    columns["ball_vx"], columns["ball_vy"] = ball_vx, ball_vy
    columns["ball_speed"] = ball_speed
    columns["ball_edge_distance"] = np.minimum.reduce([
        ball_metres[:, 0], io.PITCH[0] - ball_metres[:, 0],
        ball_metres[:, 1], io.PITCH[1] - ball_metres[:, 1]]).astype(np.float32)

    # Out of play. Coordinates are normalised to the pitch, so a ball beyond [0, 1] on either axis is
    # off the field. Measured on the public window: 27.6% of dead frames and **0.0% of live ones** --
    # a rare case where a single cheap feature is almost pure evidence (see docs/pipeline.md).
    outside_x = np.maximum(-filled[:, 0], filled[:, 0] - 1.0)
    outside_y = np.maximum(-filled[:, 1], filled[:, 1] - 1.0)
    columns["ball_outside_pitch"] = ((outside_x > 0) | (outside_y > 0)).astype(np.float32)
    columns["ball_beyond_line_m"] = np.maximum.reduce([
        outside_x * io.PITCH[0], outside_y * io.PITCH[1],
        np.zeros(n_frames)]).astype(np.float32)

    # --- ownership evidence (D2) ------------------------------------------------------------
    columns.update(ownership_evidence(metres, slot_teams, ball_metres))
    columns.update(team_shape(metres, slot_teams, ball_metres))

    # --- movement (D3) ----------------------------------------------------------------------
    # Lang et al. 2022 classify stoppages from player movement alone at 92% accuracy: during dead
    # ball, players stop, converge and walk. The tracking reports speed per slot, so this is free.
    # The ball is absent on 41% of graded frames, so a signal that does not need it is valuable.
    speed = slots[..., 2]
    tracked = ~np.isnan(speed)
    n_tracked = np.maximum(tracked.sum(axis=1), 1)
    speed_zeroed = np.where(tracked, speed, 0.0)
    columns["player_speed_mean"] = (speed_zeroed.sum(axis=1) / n_tracked).astype(np.float32)
    # Count only *tracked* slots: an untracked slot is not a stationary player, and counting it as
    # one pushes this share to ~1.0 everywhere.
    columns["player_speed_share_still"] = (
        (tracked & (speed < 1.0)).sum(axis=1) / n_tracked).astype(np.float32)
    columns["player_speed_share_moving"] = (
        (tracked & (speed > 3.0)).sum(axis=1) / n_tracked).astype(np.float32)

    # Loose ball: nobody within playing distance of it. "Controlled possession" requires a player
    # to actually have it (CHALLENGE.md), so a visible but unattended ball is evidence for D.
    d_nearest = np.minimum(columns["d_home"], columns["d_away"])
    columns["d_nearest"] = d_nearest
    columns["ball_loose"] = ((present & (d_nearest > 5.0))).astype(np.float32)

    # How long the ball has been stationary, which the instantaneous channels cannot express.
    # A free kick, a throw-in being set up or a goal kick leaves the ball still for seconds while
    # live play almost never does: measured on the public window, a ball still for >2 s with slow
    # players carries P(dead) = 0.494 against a base rate of 0.196, and covers 40% of dead frames.
    still = present & (ball_speed < 0.5)
    index = np.arange(n_frames)
    # -1 (not 0) for a still frame, so the first still frame counts 1 rather than 0.
    last_moving = np.maximum.accumulate(np.where(still, -1, index))
    columns["ball_still_frames"] = np.minimum(index - last_moving, 250).astype(np.float32)

    slow = np.clip((2.5 - columns["player_speed_mean"]) / 2.5, 0.0, 1.0)
    columns["dead_still_combo"] = (
        np.clip(columns["ball_still_frames"] / (4.0 * io.FPS), 0.0, 1.0) * slow).astype(np.float32)

    # --- tags (D3) --------------------------------------------------------------------------
    columns["in_set_piece"] = assets["in_set_piece"].astype(np.float32)
    columns["tag_team"] = assets["tag_team"].astype(np.float32)

    # Signed possession prior, accumulated in the reader over both instances of each phase (the
    # attacking team's code and the defending team's code share an interval). See io.py.
    columns["tag_possession"] = assets["tag_possession"]
    # Shaped dead-ball prior around set-piece restarts (see io.py for the shape and why).
    columns["restart_dead"] = np.asarray(assets["restart_dead"], dtype=np.float32)
    for i, name in enumerate(io.SET_PIECE_TYPES):
        columns["sp_{}".format(name.replace(" ", "_").replace("(", "").replace(")", ""))] = (
            assets["set_piece_type"] == i).astype(np.float32)
    for code, flags in sorted(assets["code_flags"].items()):
        columns["tag_{}".format(code.lower().replace(" ", "_"))] = flags.astype(np.float32)

    # --- boundaries and source quality ------------------------------------------------------
    graded = np.zeros(n_frames, dtype=bool)
    since_start = np.zeros(n_frames, dtype=np.float32)
    to_end = np.zeros(n_frames, dtype=np.float32)
    for _half, first, last in half_video_blocks(io.GAMES and "BAR-ATH"):
        graded[first:last + 1] = True
        span = np.arange(last - first + 1, dtype=np.float32)
        since_start[first:last + 1] = np.minimum(span, 2000)
        to_end[first:last + 1] = np.minimum(span[::-1], 2000)
    columns["is_graded"] = graded.astype(np.float32)
    columns["since_half_start"] = since_start
    columns["to_half_end"] = to_end
    columns["reprojection_error"] = np.clip(assets["reprojection_error"], 0.0, 20.0)

    # --- multi-scale context (D4) -----------------------------------------------------------
    for name in LAGGED_CHANNELS:
        base = np.asarray(columns[name], dtype=np.float32)
        for lag in LAGS_FRAMES:
            columns["{}_lag{}".format(name, lag)] = shift(base, lag)
            columns["{}_lead{}".format(name, lag)] = shift(base, -lag)
    for name in ("margin_top3", "d_ratio", "ball_present"):
        base = np.asarray(columns[name], dtype=np.float32)
        for half_window in ROLLING_HALVES:
            columns["{}_mean{}".format(name, half_window)] = rolling_mean(base, half_window)

    names = list(columns)
    X = np.empty((n_frames, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = np.asarray(columns[name], dtype=np.float32)

    if verbose:
        print("features: {} channels x {} frames ({:.0f} MB)"
              .format(len(names), n_frames, X.nbytes / 1e6))
        nan_columns = [n for j, n in enumerate(names) if np.isnan(X[:, j]).any()]
        print("  channels with NaN: {}".format(len(nan_columns)))
    return X, names


def save(X, names, folder=ARTIFACTS, source=None):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "features{}.npz".format("_" + source if source else ""))
    np.savez(path, X=X, names=np.array(names))
    meta = {"n_frames": int(X.shape[0]), "n_features": int(X.shape[1]),
            "names": names, "pitch": list(io.PITCH), "fps": io.FPS,
            "tau_m": TAU_M, "radii_m": list(RADII_M), "lags_frames": list(LAGS_FRAMES),
            "rolling_halves": list(ROLLING_HALVES), "source": source or "tactical"}
    with open(path.replace(".npz", "_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=1)
    return path


def load(folder=ARTIFACTS, source=None):
    path = os.path.join(folder, "features{}.npz".format("_" + source if source else ""))
    if not os.path.isfile(path):
        raise SystemExit("no features at {} -- run: python -m pipeline.features".format(path))
    data = np.load(path)
    return data["X"], [str(n) for n in data["names"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description="build the per-frame feature matrix")
    parser.add_argument("--data-dir", default=None, help="match bundle (default: $MTAH_DATA_DIR)")
    parser.add_argument("--out", default=ARTIFACTS, help="artifact folder")
    parser.add_argument("--source", default=None, help="label the artifact (default: tactical)")
    args = parser.parse_args(argv)

    X, names = build(data_dir=args.data_dir)
    path = save(X, names, args.out, args.source)
    print("wrote {}".format(os.path.relpath(path, io.ROOT)))


if __name__ == "__main__":
    main()
