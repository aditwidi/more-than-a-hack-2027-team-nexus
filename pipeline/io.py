"""Readers for the provided match assets, and the small amount of shared plumbing.

Everything here is deliberately I/O and geometry only: no decisions, no labels.

The one invariant this module protects is the submission contract -- **row i of a submission is
frame i of the tactical video** (DATA.md, "Frame alignment"). Every array returned by these
functions has exactly ``n_frames`` rows indexed by video frame, so downstream code never has to
remember that the ATD counts from 1 while Smart Tagging counts in seconds.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAMES, asset_paths, require  # noqa: E402  (path bootstrap above)

FPS = 25
PITCH = GAMES["BAR-ATH"]["pitch"]          # (105.3, 68.0) metres
N_FRAMES = GAMES["BAR-ATH"]["n_video_frames"]

# Smart Tagging is timed in seconds from the first tactical frame (DATA.md, "Time base").
# The set-piece categories we care about, in the order the Type tag reports them.
SET_PIECE_TYPES = (
    "throw-in",
    "corner kick",
    "goal kick",
    "free kick",
    "kick off (after goal)",
    "kick off (start)",
)

# Does a tag's "Team" mean the team *in possession*?
#
# +1: yes -- offensive phases, restarts, goals and post-recovery transitions are tagged with the
#     team performing them, which is the team with the ball.
# -1: no -- every DEFENDING code, and Defensive Transition, is tagged with the team *without* the
#     ball, so the sign has to be inverted. Using the raw sign pushed possession the wrong way for
#     roughly a third of the match.
#  0: ambiguous or not about possession -- ignored.
TAG_POSSESSION_SIGN = {
    "BUILD UP": 1,
    "PROGRESSION": 1,
    "CREATING CHANCES": 1,
    "BALL IN FINAL THIRD": 1,
    "BALL IN THE BOX": 1,
    "PLAYERS IN THE BOX": 1,
    "LONG BALL": 1,
    "GOALS": 1,
    "ATTACKING TRANSITION": 1,
    "SET PIECES": 1,
    "DEFENDING IN ATTACKING THIRD": -1,
    "DEFENDING IN MIDDLE THIRD": -1,
    "DEFENDING IN DEFENSIVE THIRD": -1,
    "DEFENSIVE TRANSITION": -1,
}


def atd_paths(data_dir=None):
    return require(["atd_meta", "atd_raw"], data_dir=data_dir)


def load_atd(raw_path, n_frames=N_FRAMES):
    """Parse the raw ATD export into per-frame player slots and ball position.

    Line format is ``<frame>:<x,y,s>;<x,y,s>;...:<x,y,s>`` -- 46 player slots with a speed channel,
    then the ball. Missing observations are the literal ``NaN``. The whole file is 28.3M numbers
    (~226 MB of float64), so it is flattened and reshaped in one pass rather than parsed per line:
    3 s instead of several minutes.

    Returns
    -------
    slots : (n_frames, n_slots, 3) float32
        x, y in normalised pitch coordinates and speed in m/s, NaN where untracked.
    ball : (n_frames, 3) float32
        x, y, speed. NaN where the ball is not visible -- 43% of all frames.
    """
    with open(raw_path) as handle:
        text = handle.read()
    for delimiter in (":", ";", ","):
        text = text.replace(delimiter, " ")
    text = text.replace("\n", " ")

    values = np.fromstring(text, sep=" ", dtype=np.float64)
    del text

    n_slots = (values.size // n_frames - 4) // 3
    if n_slots * 3 * n_frames + 4 * n_frames != values.size:
        raise ValueError(
            "unexpected ATD layout: {} values for {} frames ({} slots detected)"
            .format(values.size, n_frames, n_slots))

    table = values.reshape(n_frames, 1 + n_slots * 3 + 3)
    del values

    # The file counts frames from 1; submissions count from 0. Assert rather than assume, because
    # an off-by-one here is invisible downstream and fatal at scoring time.
    frame_ids = table[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("ATD frame ids are not 1..{} -- check the export, not the offset"
                         .format(n_frames))

    slots = table[:, 1:1 + n_slots * 3].reshape(n_frames, n_slots, 3).astype(np.float32)
    ball = table[:, 1 + n_slots * 3:].astype(np.float32)
    return slots, ball


def track_teams(meta_path):
    """Team code per track slot: 0 = home (Barcelona), 1 = away (Athletic).

    Slots are tracklets, not people (DATA.md, "What it does not contain"): 46 slots for 22 players,
    appearing, vanishing and swapping identity. The team assignment is the one stable thing about
    them, and it is what the FIFA/EPTS metadata actually guarantees.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    home_id = team_ids[0]           # first team in the file is the home side (EPTS convention)
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([0 if team_id == home_id else 1 for team_id in slots], dtype=np.int8)


def load_homography(path, n_frames=N_FRAMES):
    """Reprojection error and validity per frame, as (n_frames,) float32 / bool arrays.

    Only the two scalars are needed as features. The 3x3 matrices are 1.7 MB per source and are
    only required when projecting to pixels, which this pipeline does not do.
    """
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8")
    if table.size != n_frames:
        raise ValueError("{}: {} rows for {} video frames".format(path, table.size, n_frames))
    order = np.argsort(table["frame"])
    error = np.asarray(table["reprojection_error"][order], dtype=np.float32)
    valid = np.asarray(table["valid"][order], dtype=bool)
    return error, valid


def load_phases(path, n_frames=N_FRAMES):
    """Every Smart Tagging instance as a dict: code, team sign, type, half, frame range.

    Separate from the per-frame arrays because verification needs the *phases* -- "at the start of
    this Build Up, was the tagged team actually the one we labelled?" -- which an aggregated array
    cannot express.
    """
    phases = []
    root = ET.parse(path).getroot()
    for instance in root.iter("instance"):
        start = _seconds(instance.findtext("start"))
        end = _seconds(instance.findtext("end"))
        if start is None:
            continue
        labels = {}
        for label in instance.iter("label"):
            group = (label.findtext("group") or "").strip()
            value = (label.findtext("text") or "").strip()
            if group:
                labels[group] = value

        team = labels.get("Team", "").lower()
        if team.startswith("barcelona"):
            sign = 1
        elif team.startswith("atletic") or team.startswith("athletic"):
            sign = -1
        else:
            sign = 0

        first = max(0, int(round(start * FPS)))
        last = min(n_frames - 1, int(round((end if end is not None else start) * FPS)))
        if last < first:
            continue
        phases.append({"code": (instance.findtext("code") or "").strip().upper(),
                       "team": sign, "type": labels.get("Type", ""),
                       "half": labels.get("Half", ""),
                       "first": first, "last": last})
    return phases


def load_smart_tagging(path, n_frames=N_FRAMES):
    """Smart Tagging as per-frame flags.

    Returns a dict of (n_frames,) arrays:

    ``in_set_piece``   bool -- frame lies inside a SET PIECES interval
    ``set_piece_type`` int8 -- index into SET_PIECE_TYPES, -1 outside a set piece
    ``tag_team``       int8 -- +1 home, -1 away, 0 when the tag is N/A (all throw-ins are)
    ``code_flags``     dict of code -> bool array, one entry per Smart Tagging code
    ``code_instances`` dict of code -> number of instances in the file
    ``tag_possession`` float32 -- signed possession prior in [-1, 1], +1 home, -1 away
    ``restart_dead``   float32 -- [0, 1] dead-ball prior shaped around set-piece restarts
    ``meta``           dict of tag group -> {value: instance count}, for verification against DATA.md

    Tags are nested as ``<label><group>Team</group><text>Barcelona</text></label>``, so they have to
    be read as group/value pairs; ``findtext("Team")`` returns nothing.

    A phase is recorded **twice**, once with the attacking team's code and once with the defending
    team's code over the same interval (PROGRESSION and DEFENDING IN MIDDLE THIRD have identical
    frame counts, as do ATTACKING and DEFENSIVE TRANSITION). So a single per-frame "Team" value is
    meaningless -- whichever instance was written last wins. ``tag_possession`` accumulates
    ``sign(code) * team`` over both instances of the phase, which makes the pair agree, and averages.
    """
    root = ET.parse(path).getroot()
    in_sp = np.zeros(n_frames, dtype=bool)
    sp_type = np.full(n_frames, -1, dtype=np.int8)
    tag_team = np.zeros(n_frames, dtype=np.int8)
    signed = np.zeros(n_frames, dtype=np.float32)
    votes = np.zeros(n_frames, dtype=np.int16)
    restart_dead = np.zeros(n_frames, dtype=np.float32)
    code_flags = {}
    code_instances = {}
    meta = {}

    type_index = {name: i for i, name in enumerate(SET_PIECE_TYPES)}

    for instance in root.iter("instance"):
        start = _seconds(instance.findtext("start"))
        end = _seconds(instance.findtext("end"))
        if start is None:
            continue
        first = int(round(start * FPS))
        last = int(round((end if end is not None else start) * FPS))
        first, last = max(0, first), min(n_frames - 1, last)
        if last < first:
            continue

        code = (instance.findtext("code") or "").strip().upper()
        labels = {}
        for label in instance.iter("label"):
            group = (label.findtext("group") or "").strip()
            value = (label.findtext("text") or "").strip()
            if group:
                labels[group] = value
                meta.setdefault(group, {})
                meta[group][value] = meta[group].get(value, 0) + 1

        team = labels.get("Team", "").lower()
        if team.startswith("barcelona"):
            value = 1
        elif team.startswith("atletic") or team.startswith("athletic"):
            value = -1
        else:
            value = 0
        tag_team[first:last + 1] = value
        sign = TAG_POSSESSION_SIGN.get(code, 0)
        if sign and value:
            signed[first:last + 1] += sign * value
            votes[first:last + 1] += 1
        code_flags.setdefault(code, np.zeros(n_frames, dtype=bool))[first:last + 1] = True
        code_instances[code] = code_instances.get(code, 0) + 1

        if code == "SET PIECES":
            in_sp[first:last + 1] = True
            label = labels.get("Type", "").lower()
            if label in type_index:
                sp_type[first:last + 1] = type_index[label]

            # Extend backwards: the ball is already dead while it is being retrieved.
            before = int(round(RESTART_DEAD_BEFORE_S * FPS))
            span_first = max(0, first - before)
            span_last = last
            length_s = max((last - first + 1), 1) / float(FPS)
            offsets = (np.arange(span_first, span_last + 1) - first) / float(FPS)
            restart_dead[span_first:span_last + 1] = np.maximum(
                restart_dead[span_first:span_last + 1],
                _restart_kernel(offsets, length_s))

    tag_possession = np.where(votes > 0, signed / np.maximum(votes, 1), 0.0).astype(np.float32)
    return {"in_set_piece": in_sp, "set_piece_type": sp_type, "tag_team": tag_team,
            "tag_possession": tag_possession, "restart_dead": restart_dead,
            "code_flags": code_flags, "code_instances": code_instances, "meta": meta}


def _seconds(text):
    """Smart Tagging timecodes are plain seconds (DATA.md, "Time base"), but tolerate hh:mm:ss."""
    if not text:
        return None
    text = text.strip()
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    total = 0.0
    for part in text.split(":"):
        total = total * 60.0 + float(part)
    return total


# Dead-ball prior around a restart. A tagged SET PIECES interval spans the whole restart sequence --
# ball retrieved, players set, the restart struck, play resumed -- and the strike is not at a fixed
# offset from the start (the Kick Off (Start) instance runs 443.88-448.88 s for a kickoff struck at
# 447.44 s, i.e. 3.5 s in). So the prior is shaped by *fraction of the interval*: certainly dead at
# the start and through the set-up, decaying to nothing by the end, where play has resumed.
RESTART_DEAD_BEFORE_S = 1.5     # frames just before the tagged start are dead too
RESTART_DEAD_HOLD_FRACTION = 0.4  # fraction of the interval that is certainly dead


def _restart_kernel(offset_s, length_s):
    """Dead-ball prior for frames at ``offset_s`` from the start of an interval of ``length_s``."""
    offset_s = np.asarray(offset_s, dtype=np.float32)
    length_s = max(float(length_s), 1e-3)
    fraction = offset_s / length_s
    tail = np.clip((1.0 - fraction) / (1.0 - RESTART_DEAD_HOLD_FRACTION), 0.0, 1.0)
    value = np.where(fraction <= RESTART_DEAD_HOLD_FRACTION, 1.0, tail)
    return np.where(offset_s < 0, 1.0, value).astype(np.float32)


def load_all(data_dir=None):
    """Everything Step 1 needs, as a single dict of arrays on the video-frame timeline."""
    meta_path, raw_path = atd_paths(data_dir)
    paths = asset_paths(data_dir=data_dir)

    slots, ball = load_atd(raw_path)
    teams = track_teams(meta_path)
    error, valid = load_homography(paths["homography"])
    tags = load_smart_tagging(paths["smart_tagging"])

    out = {"slots": slots, "ball": ball, "slot_teams": teams,
           "reprojection_error": error, "homography_valid": valid}
    out.update(tags)
    return out
