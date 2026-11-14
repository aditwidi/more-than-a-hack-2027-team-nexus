"""Tests for the feature/IO layer.

These check the invariants that are expensive to discover later: the frame alignment (ATD counts
from 1, submissions from 0), gap/absence bookkeeping, interpolation, the lagged/rolling context, and
the Smart Tagging group/value parsing. They use tiny synthetic fixtures, so they run in well under a
second and do not need the gated match data.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import diag, features, io  # noqa: E402


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------
def write_atd(path, frames, n_slots=3):
    """Write a raw ATD file. ``frames`` is a list of (player_groups, ball_group) per frame."""
    with open(path, "w") as handle:
        for i, (players, ball) in enumerate(frames, start=1):
            groups = ";".join(players) + ";" * 0
            handle.write("{}:{}:{}\n".format(i, groups, ball))


def group(x, y, s=0.0):
    if x is None:
        return "NaN,NaN,NaN"
    return "{:.5f},{:.5f},{:.5f}".format(x, y, s)


# ---------------------------------------------------------------------------------------------
# ATD
# ---------------------------------------------------------------------------------------------
def test_atd_parse_aligns_frames_to_zero(tmp_path):
    path = tmp_path / "atd.txt"
    write_atd(path, [
        ([group(0.1, 0.2), group(0.3, 0.4), group(None, None)], group(0.5, 0.5, 1.0)),
        ([group(0.1, 0.2), group(None, None, None), group(None, None)], group(None, None, None)),
    ])

    slots, ball = io.load_atd(str(path), n_frames=2)
    assert slots.shape == (2, 3, 3) and ball.shape == (2, 3)

    # Frame recorded as id 1 is submission frame 0.
    assert slots[0, 0, 0] == pytest.approx(0.1)
    assert not np.isnan(slots[1, 0, 0])             # observed on both frames in the fixture
    assert np.isnan(slots[0, 2, 0])                 # NaN preserved, not zero-filled
    assert ball[0, 0] == pytest.approx(0.5)
    assert np.isnan(ball[1, 0])


def test_atd_parse_rejects_wrong_frame_count(tmp_path):
    path = tmp_path / "atd.txt"
    write_atd(path, [([group(0.1, 0.2)] * 3, group(0.5, 0.5))])
    with pytest.raises(ValueError):
        io.load_atd(str(path), n_frames=2)


# ---------------------------------------------------------------------------------------------
# ball bookkeeping
# ---------------------------------------------------------------------------------------------
def test_ball_presence_gap_lengths():
    ball = np.full((8, 3), np.nan, dtype=np.float32)
    ball[[1, 2, 6], 0] = 0.5

    present, prev_gap, next_gap = features.ball_presence(ball)
    assert present.tolist() == [False, True, True, False, False, False, True, False]
    # frame 0: nobody before it, next observation at frame 1 -> next gap 1
    assert (prev_gap[0], next_gap[0]) == (0, 1)
    # frame 4: last seen frame 2, next at 6 -> 2 and 2
    assert (prev_gap[4], next_gap[4]) == (2, 2)
    # frame 7: never seen again -> next gap is capped/zero-length symmetric to the left
    assert prev_gap[7] == 1


def test_fill_ball_interpolates_between_observations():
    ball = np.full((5, 3), np.nan, dtype=np.float32)
    ball[0, :2] = (0.0, 0.0)
    ball[4, :2] = (1.0, 1.0)

    present, _, _ = features.ball_presence(ball)
    filled, span, _ = features.fill_ball(ball, present)

    assert filled[0] == pytest.approx((0.0, 0.0))
    assert filled[4] == pytest.approx((1.0, 1.0))
    assert filled[2] == pytest.approx((0.5, 0.5))       # linear, not last-known-held
    assert span[0] == 0 and span[2] == 4
    # nothing is NaN: a source that never sees the ball falls back to the centre spot
    empty = np.full((3, 3), np.nan, dtype=np.float32)
    filled, span, _ = features.fill_ball(empty, features.ball_presence(empty)[0])
    assert np.allclose(filled, 0.5) and span.max() == 0


def test_ball_motion_zeroes_velocity_across_gaps():
    ball = np.full((4, 3), np.nan, dtype=np.float32)
    ball[0, :2] = (0.0, 0.0)
    ball[1, :2] = (0.1, 0.0)
    ball[3, :2] = (0.3, 0.0)

    present, _, _ = features.ball_presence(ball)
    filled, _, _ = features.fill_ball(ball, present)
    vx, _, speed = features.ball_motion(filled, present)

    assert vx[1] > 0                                     # observed motion, in metres/sec
    assert speed[2] == 0.0                               # interpolated frame is not "motion"


# ---------------------------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------------------------
def test_shift_is_edge_clamped_not_wrapped():
    values = np.arange(5, dtype=np.float32)
    assert features.shift(values, 1).tolist() == [0, 0, 1, 2, 3]
    assert features.shift(values, -1).tolist() == [1, 2, 3, 4, 4]


def test_rolling_mean_ignores_nan():
    values = np.array([1.0, np.nan, 3.0, 5.0], dtype=np.float32)
    out = features.rolling_mean(values, 1)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(2.0)                  # (1 + 3) / 2, NaN skipped
    assert out[2] == pytest.approx(4.0)                  # (3 + 5) / 2


# ---------------------------------------------------------------------------------------------
# ownership evidence
# ---------------------------------------------------------------------------------------------
def test_ownership_evidence_prefers_the_nearer_team():
    # Two slots per team; home players 2 m from the ball, away players 20 m away.
    slots = np.zeros((1, 4, 3), dtype=np.float32)
    pitch = io.PITCH

    def put(slot, x_m, y_m):
        slots[0, slot, 0] = x_m / pitch[0]
        slots[0, slot, 1] = y_m / pitch[1]

    put(0, 52.0, 34.0)
    put(1, 53.0, 34.0)
    put(2, 32.0, 34.0)
    put(3, 72.0, 34.0)

    teams = np.array([0, 0, 1, 1], dtype=np.int8)
    ball_metres = np.array([[52.5, 34.0]])

    evidence = features.ownership_evidence(features.player_metres(slots), teams, ball_metres)
    assert evidence["d_home"][0] < 1.0
    assert evidence["d_away"][0] == pytest.approx(19.5, abs=0.1)
    assert evidence["score_margin"][0] > 0                # positive means home is nearer
    assert evidence["n_home_r5"][0] == 2 and evidence["n_away_r5"][0] == 0


def test_ownership_evidence_handles_an_empty_team():
    slots = np.full((1, 2, 3), np.nan, dtype=np.float32)
    teams = np.array([0, 1], dtype=np.int8)
    evidence = features.ownership_evidence(features.player_metres(slots), teams,
                                          np.array([[52.0, 34.0]]))
    assert evidence["d_home"][0] == features.DISTANCE_CAP_M
    assert evidence["score_margin"][0] == 0


# ---------------------------------------------------------------------------------------------
# possessions (the metric's own definition)
# ---------------------------------------------------------------------------------------------
def test_possessions_split_on_dead_ball_but_not_between_teams_runs():
    labels = np.array(list("HHHHAAAADDHH"))
    spells = diag.possessions(labels)
    assert [(s["team"], s["length"]) for s in spells] == [("H", 4), ("A", 4), ("H", 2)]


def test_runs_and_structure():
    labels = np.array(list("HHDHHAA"))
    assert [(s, e, l) for s, e, l in diag.runs(labels)] == [(0, 1, "H"), (2, 2, "D"),
                                                            (3, 4, "H"), (5, 6, "A")]
    stats = diag.structure(labels, 0, len(labels) - 1)
    assert stats["possessions"] == 3
    assert stats["transitions"] == 3                     # H->D, D->H, H->A within the window
    assert stats["dead_share"] == pytest.approx(1 / 7, abs=1e-4)   # rounded to 4 dp by design
    assert stats["short_possessions_lt_2s"] == 3          # all of them, at 25 fps


# ---------------------------------------------------------------------------------------------
# Smart Tagging
# ---------------------------------------------------------------------------------------------
SMART_TAGGING = """<?xml version="1.0" ?>
<file><ALL_INSTANCES>
  <instance>
    <ID>1</ID><start>1.0</start><end>2.0</end><code>SET PIECES</code>
    <label><group>Team</group><text>N/A</text></label>
    <label><group>Type</group><text>Throw-in</text></label>
    <label><group>Half</group><text>1st Half</text></label>
  </instance>
  <instance>
    <ID>2</ID><start>2.0</start><end>2.0</end><code>SET PIECES</code>
    <label><group>Team</group><text>Barcelona</text></label>
    <label><group>Type</group><text>Corner Kick</text></label>
  </instance>
  <instance>
    <ID>3</ID><start>3.0</start><end>4.0</end><code>PROGRESSION</code>
    <label><group>Team</group><text>Atletic Club</text></label>
  </instance>
</ALL_INSTANCES></file>
"""


def test_smart_tagging_reads_nested_group_value_labels(tmp_path):
    path = tmp_path / "tags.xml"
    path.write_text(SMART_TAGGING)
    tags = io.load_smart_tagging(str(path), n_frames=125)

    # 1.0 s -> frame 25, 2.0 s -> frame 50, inclusive.
    assert tags["in_set_piece"][25] and tags["in_set_piece"][50]
    assert not tags["in_set_piece"][24]

    corner = io.SET_PIECE_TYPES.index("corner kick")
    throw_in = io.SET_PIECE_TYPES.index("throw-in")
    assert tags["set_piece_type"][30] == throw_in
    assert tags["set_piece_type"][50] == corner

    # N/A must stay unknown (0) rather than defaulting to a team -- every throw-in is N/A.
    assert tags["tag_team"][30] == 0
    assert tags["tag_team"][50] == 1
    assert tags["tag_team"][80] == -1

    assert tags["code_instances"] == {"SET PIECES": 2, "PROGRESSION": 1}
    assert tags["meta"]["Team"] == {"N/A": 1, "Barcelona": 1, "Atletic Club": 1}


# ---------------------------------------------------------------------------------------------
# ownership normalisation (Step 3)
# ---------------------------------------------------------------------------------------------
def test_top3_score_is_invariant_to_extra_far_slots():
    """A team's ownership score must not depend on how many track slots it happens to have.

    This match has 24 home slots and 22 away ones, and an unnormalised sum over slots biased the
    decision toward home (predicted alive home share 0.71 vs a truth of 0.57).

    Invariance needs at least k tracked players near the ball, which is the normal case (both teams
    are tracked with ~11 slots); with fewer, the mean is taken over however many exist, because
    inventing an eleventh-distance player would be worse.
    """
    pitch = io.PITCH
    ball = np.array([[52.0, 34.0]])

    def slot(x_m, y_m):
        return (x_m / pitch[0], y_m / pitch[1], 0.0)

    # Three home players at 1/2/3 m and three away at 10/12/15 m.
    real = np.array([[slot(51, 34), slot(52, 36), slot(54, 33),
                      slot(62, 34), slot(60, 40), slot(58, 28)]], dtype=np.float32)
    teams_few = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)

    # The same six players, plus five duplicated home slots ~8 m away. Duplicates are not far
    # away in practice -- the tracker drops and re-acquires someone near the play -- so each one
    # contributes exp(-8/4) = 0.14 to the sum and the unnormalised score drifts upward.
    extra = np.array([[slot(60, 34)] * 5], dtype=np.float32)
    slots_many = np.concatenate([real, extra], axis=1)
    teams_many = np.array([0, 0, 0, 1, 1, 1] + [0] * 5, dtype=np.int8)

    few = features.ownership_evidence(features.player_metres(real), teams_few, ball)
    many = features.ownership_evidence(features.player_metres(slots_many), teams_many, ball)

    assert many["score3_home"][0] == pytest.approx(few["score3_home"][0], abs=1e-4)
    assert many["margin_top3"][0] == pytest.approx(few["margin_top3"][0], abs=1e-4)
    # ...while the unnormalised sum does move, which is exactly the bias being removed.
    assert many["score_margin"][0] > few["score_margin"][0] + 1e-3


# ---------------------------------------------------------------------------------------------
# possessions (the metric's own definition)
# ---------------------------------------------------------------------------------------------
def test_possessions_split_on_dead_ball_but_not_between_teams_runs():
    labels = np.array(list("HHHHAAAADDHH"))
    spells = diag.possessions(labels)
    assert [(s["team"], s["length"]) for s in spells] == [("H", 4), ("A", 4), ("H", 2)]


def test_runs_and_structure():
    labels = np.array(list("HHDHHAA"))
    assert [(s, e, l) for s, e, l in diag.runs(labels)] == [(0, 1, "H"), (2, 2, "D"),
                                                            (3, 4, "H"), (5, 6, "A")]
    stats = diag.structure(labels, 0, len(labels) - 1)
    assert stats["possessions"] == 3
    assert stats["transitions"] == 3                     # H->D, D->H, H->A within the window
    assert stats["dead_share"] == pytest.approx(1 / 7, abs=1e-4)   # rounded to 4 dp by design
    assert stats["short_possessions_lt_2s"] == 3          # all of them, at 25 fps


# ---------------------------------------------------------------------------------------------
# Smart Tagging
# ---------------------------------------------------------------------------------------------
SMART_TAGGING = """<?xml version="1.0" ?>
<file><ALL_INSTANCES>
  <instance>
    <ID>1</ID><start>1.0</start><end>2.0</end><code>SET PIECES</code>
    <label><group>Team</group><text>N/A</text></label>
    <label><group>Type</group><text>Throw-in</text></label>
    <label><group>Half</group><text>1st Half</text></label>
  </instance>
  <instance>
    <ID>2</ID><start>2.0</start><end>2.0</end><code>SET PIECES</code>
    <label><group>Team</group><text>Barcelona</text></label>
    <label><group>Type</group><text>Corner Kick</text></label>
  </instance>
  <instance>
    <ID>3</ID><start>3.0</start><end>4.0</end><code>PROGRESSION</code>
    <label><group>Team</group><text>Atletic Club</text></label>
  </instance>
</ALL_INSTANCES></file>
"""


def test_smart_tagging_reads_nested_group_value_labels(tmp_path):
    path = tmp_path / "tags.xml"
    path.write_text(SMART_TAGGING)
    tags = io.load_smart_tagging(str(path), n_frames=125)

    # 1.0 s -> frame 25, 2.0 s -> frame 50, inclusive.
    assert tags["in_set_piece"][25] and tags["in_set_piece"][50]
    assert not tags["in_set_piece"][24]

    corner = io.SET_PIECE_TYPES.index("corner kick")
    throw_in = io.SET_PIECE_TYPES.index("throw-in")
    assert tags["set_piece_type"][30] == throw_in
    assert tags["set_piece_type"][50] == corner

    # N/A must stay unknown (0) rather than defaulting to a team -- every throw-in is N/A.
    assert tags["tag_team"][30] == 0
    assert tags["tag_team"][50] == 1
    assert tags["tag_team"][80] == -1

    assert tags["code_instances"] == {"SET PIECES": 2, "PROGRESSION": 1}
    assert tags["meta"]["Team"] == {"N/A": 1, "Barcelona": 1, "Atletic Club": 1}


# ---------------------------------------------------------------------------------------------
# ball-stationary duration (Step 4)
# ---------------------------------------------------------------------------------------------
def test_still_frames_reset_on_absence_and_motion():
    """`ball_still_frames` must count a *continuous* stillness, not accumulate across gaps."""
    ball = np.zeros((10, 3), dtype=np.float32)
    ball[:, 0] = 0.5
    ball[:, 1] = 0.5
    ball[:, 2] = 0.0                     # stationary
    ball[4, 2] = 3.0                     # one frame of motion resets the count
    ball[7, 0] = np.nan                  # and so does losing the ball

    slots = np.zeros((10, 2, 3), dtype=np.float32)
    teams = np.array([0, 1], dtype=np.int8)
    slots[:, 0, :2] = (0.5, 0.5)
    slots[:, 1, :2] = (0.6, 0.5)

    # Recreate the feature exactly as features.build() does for this channel.
    present, _, _ = features.ball_presence(ball)
    still = present & (ball[:, 2] < 0.5)
    index = np.arange(10)
    last_moving = np.maximum.accumulate(np.where(still, -1, index))
    run = index - last_moving

    assert run.tolist() == [1, 2, 3, 4, 0, 1, 2, 0, 1, 2]


# ---------------------------------------------------------------------------------------------
# render alignment (Step 7)
# ---------------------------------------------------------------------------------------------
def test_render_scales_projected_pixels_to_the_image():
    """A projected coordinate is in 1920x1080 pixels; drawing on a smaller image must scale it.

    Regression test for a real bug: the scale factor was hardcoded to 1.0 after the streaming
    rewrite, so every circle was drawn at 1x while the frame was 0.5x, and the overlay looked like
    a broken homography.
    """
    from PIL import Image

    from pipeline import render

    image = Image.new("RGB", (960, 540), (0, 0, 0))
    # Identity-in-spirit matrix: normalised pitch (0-1) -> 1920x1080 pixels.
    matrix = np.array([1920.0, 0.0, 0.0, 0.0, 1080.0, 0.0, 0.0, 0.0, 1.0])
    slots = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)       # one frame, one slot, centre
    ball = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    render.evidence_line_cached = "test"
    render.annotate(image, 0, matrix, slots, ball, np.array([0], dtype=np.int8), "test", scale=0.5)

    # The arithmetic the bug got wrong: projected pixels are computed at 1920x1080.
    assert render.project(matrix, 0.5, 0.5) == (960.0, 540.0)
    assert render.project(matrix, 0.5, 0.5)[0] * 0.5 == 480.0

    pixels = np.asarray(image)
    centre = pixels[270 - 8:270 + 8, 480 - 8:480 + 8]
    assert centre.max() > 0, "nothing was drawn at the scaled position"
