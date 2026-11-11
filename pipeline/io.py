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


def load_smart_tagging(path, n_frames=N_FRAMES):
    """Smart Tagging as per-frame flags.

    Returns a dict of (n_frames,) arrays:

    ``in_set_piece``   bool -- frame lies inside a SET PIECES interval
    ``set_piece_type`` int8 -- index into SET_PIECE_TYPES, -1 outside a set piece
    ``tag_team``       int8 -- +1 home, -1 away, 0 when the tag is N/A (all throw-ins are)
    ``code_flags``     dict of code -> bool array, one entry per Smart Tagging code
    ``code_instances`` dict of code -> number of instances in the file
    ``meta``           dict of tag group -> {value: instance count}, for verification against DATA.md

    Tags are nested as ``<label><group>Team</group><text>Barcelona</text></label>``, so they have to
    be read as group/value pairs; ``findtext("Team")`` returns nothing.
    """
    root = ET.parse(path).getroot()
    in_sp = np.zeros(n_frames, dtype=bool)
    sp_type = np.full(n_frames, -1, dtype=np.int8)
    tag_team = np.zeros(n_frames, dtype=np.int8)
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
        code_flags.setdefault(code, np.zeros(n_frames, dtype=bool))[first:last + 1] = True
        code_instances[code] = code_instances.get(code, 0) + 1

        if code == "SET PIECES":
            in_sp[first:last + 1] = True
            label = labels.get("Type", "").lower()
            if label in type_index:
                sp_type[first:last + 1] = type_index[label]

    return {"in_set_piece": in_sp, "set_piece_type": sp_type, "tag_team": tag_team,
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
