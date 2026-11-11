"""Tests for emissions and decoding.

The decisive properties are structural, not numeric: a single noisy frame must not create a
possession, a short ball loss must not split one, and a real stoppage must. Those are asserted
directly, so the decoder can be refactored without quietly losing the behaviour the ranking metric
depends on.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import decode, emissions, io  # noqa: E402

N_FRAMES = 400


def make_features(n_frames=N_FRAMES, **overrides):
    """A minimal feature matrix with every required channel, all neutral."""
    names = ["ball_present", "ball_gap_prev", "ball_gap_next", "ball_fill_span",
             "in_set_piece", "score_margin", "tag_team", "is_graded",
             "d_home", "d_away", "ball_outside_pitch",
             "player_speed_mean", "player_speed_share_moving", "ball_loose"]
    X = np.zeros((n_frames, len(names)), dtype=np.float32)
    column = {name: j for j, name in enumerate(names)}

    X[:, column["ball_present"]] = 1.0
    X[:, column["is_graded"]] = 1.0
    X[:, column["d_home"]] = 5.0
    X[:, column["d_away"]] = 5.0
    X[:, column["player_speed_mean"]] = 2.5
    X[:, column["player_speed_share_moving"]] = 0.3

    for name, value in overrides.items():
        X[:, column[name]] = value
    return X, names


# ---------------------------------------------------------------------------------------------
# emissions
# ---------------------------------------------------------------------------------------------
def test_emission_logits_shape_and_required_channels():
    X, names = make_features()
    logits = emissions.emission_logits(X, names)
    assert logits.shape == (N_FRAMES, 3)

    with pytest.raises(KeyError):
        emissions.emission_logits(X[:, :5], names[:5])


def test_emission_prefers_home_when_margin_is_positive():
    X, names = make_features(score_margin=1.0)
    logits = emissions.emission_logits(X, names)
    assert logits[:, emissions.HOME].mean() > logits[:, emissions.AWAY].mean()

    X, names = make_features(score_margin=-1.0)
    logits = emissions.emission_logits(X, names)
    assert logits[:, emissions.AWAY].mean() > logits[:, emissions.HOME].mean()


def test_out_of_play_ball_raises_the_dead_score():
    neutral, names = make_features()
    outside, _ = make_features(ball_outside_pitch=1.0)
    assert (emissions.emission_logits(outside, names)[:, emissions.DEAD]
            > emissions.emission_logits(neutral, names)[:, emissions.DEAD]).all()


def test_long_ball_absence_raises_the_dead_score():
    short, names = make_features(ball_present=0.0, ball_gap_prev=2, ball_gap_next=2)
    long, _ = make_features(ball_present=0.0, ball_gap_prev=125, ball_gap_next=125)
    assert (emissions.emission_logits(long, names)[:, emissions.DEAD]
            > emissions.emission_logits(short, names)[:, emissions.DEAD]).all()


def test_still_players_raise_the_dead_score():
    moving, names = make_features(player_speed_mean=3.5, player_speed_share_moving=0.6)
    still, _ = make_features(player_speed_mean=0.4, player_speed_share_moving=0.0)
    assert (emissions.emission_logits(still, names)[:, emissions.DEAD]
            > emissions.emission_logits(moving, names)[:, emissions.DEAD]).all()


def test_pre_kickoff_block_is_forced_dead():
    """FAQ.md publishes that the 2,934 frames before the H2 kickoff are all dead ball."""
    X, names = make_features(io.N_FRAMES)
    logits = emissions.emission_logits(X, names)
    _, second_first, _ = __import__("games").half_video_blocks("BAR-ATH")[1]
    frame = second_first + 10
    assert logits[frame, emissions.DEAD] > logits[frame, emissions.HOME]
    assert logits[frame, emissions.DEAD] > logits[frame, emissions.AWAY]


# ---------------------------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------------------------
def test_viterbi_follows_strong_evidence():
    logits = np.zeros((50, 3), dtype=np.float32)
    logits[:, emissions.AWAY] = 10.0
    params = dict(emissions.DEFAULTS)
    assert (decode.viterbi(logits, params) == emissions.AWAY).all()


def test_viterbi_ignores_a_single_noisy_frame():
    """One frame of opposing evidence must not create a possession.

    A possession is worth far more than one frame in the ranking metric, so this is the behaviour
    that matters: the switch costs more than the frame gains. Note the path needs an established
    state first -- the initial state carries no prior, so flat evidence may start anywhere.
    """
    logits = np.zeros((100, 3), dtype=np.float32)
    logits[:, emissions.HOME] = 2.0
    logits[50, emissions.AWAY] = 4.0                 # one bad read, 4 > 2 but < 2 + switch cost
    params = dict(emissions.DEFAULTS)
    params["switch_penalty"] = 6.0
    states = decode.viterbi(logits, params)
    assert set(states.tolist()) == {emissions.HOME}


def test_viterbi_switches_on_sustained_evidence():
    logits = np.zeros((200, 3), dtype=np.float32)
    logits[:100, emissions.HOME] = 2.0
    logits[100:, emissions.AWAY] = 2.0                # sustained, not a blip
    params = dict(emissions.DEFAULTS)
    states = decode.viterbi(logits, params)
    assert states[0] == emissions.HOME and states[-1] == emissions.AWAY
    assert len(decode.runs(states)) == 2              # exactly one transition


# ---------------------------------------------------------------------------------------------
# structural cleanup
# ---------------------------------------------------------------------------------------------
def params(**overrides):
    p = dict(emissions.DEFAULTS)
    p.update({"min_possession_frames": 50, "min_dead_frames": 25})
    p.update(overrides)
    return p


def test_cleanup_absorbs_a_short_ball_loss_inside_a_possession():
    states = np.array([0] * 100 + [2] * 10 + [0] * 100, dtype=np.int8)
    assert set(decode.cleanup(states, params()).tolist()) == {0}


def test_cleanup_keeps_a_real_stoppage_as_a_break():
    """`H H D D H H` is two possessions, not one -- so a long D must survive."""
    states = np.array([0] * 100 + [2] * 60 + [0] * 100, dtype=np.int8)
    cleaned = decode.cleanup(states, params())
    assert set(cleaned.tolist()) == {0, 2}
    assert len(decode.runs(cleaned)) == 3


def test_cleanup_absorbs_a_short_team_blip_inside_a_stoppage():
    states = np.array([2] * 100 + [1] * 10 + [2] * 100, dtype=np.int8)
    assert set(decode.cleanup(states, params()).tolist()) == {2}


def test_cleanup_absorbs_flicker_between_same_team():
    states = np.array([0] * 100 + [1] * 5 + [0] * 100, dtype=np.int8)
    assert set(decode.cleanup(states, params()).tolist()) == {0}


def test_cleanup_does_not_merge_across_dead_ball():
    """Same team either side of a long stoppage stays two possessions."""
    states = np.array([0] * 60 + [2] * 60 + [0] * 60, dtype=np.int8)
    assert len(decode.runs(decode.cleanup(states, params()))) == 3


# ---------------------------------------------------------------------------------------------
# full decode
# ---------------------------------------------------------------------------------------------
def test_decode_marks_ungraded_frames_as_no_answer():
    X, names = make_features(is_graded=0.0)
    labels = decode.decode(X, names)
    assert set(labels.tolist()) == {"X"}


def test_decode_produces_only_valid_labels():
    X, names = make_features()
    labels = decode.decode(X, names)
    assert set(labels.tolist()) <= set("HADX")
    assert len(labels) == N_FRAMES


def test_submission_roundtrip(tmp_path):
    labels = np.array(list("HADX") * 3, dtype="<U1")
    path = decode.write_submission(str(tmp_path / "sub.csv"), labels)

    with open(path) as handle:
        lines = handle.read().splitlines()
    assert lines[0] == "frame,label"
    assert lines[1] == "0,H"
    assert len(lines) == len(labels) + 1
