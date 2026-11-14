"""Render annotated frames: project the tracking onto the video and burn in our own labels.

This is the visual half of verification (the quantitative half is `pipeline/verify.py`). It answers
questions the numbers cannot: *does the predicted label match what is on the pitch in this frame*,
and *why did the model think so*.

How it works: for each requested frame, take the homography row for that frame, project every tracked
slot (normalised pitch coordinates, the same convention the ATD uses) into image pixels, draw players
by team and the ball, and print the label plus the evidence the emission used.

    # a contact sheet around a specific moment (frame numbers are submission rows)
    python -m pipeline.render --start 45900 --seconds 6 --every 25 --out artifacts/goal.png

    # an annotated clip you can actually watch
    python -m pipeline.render --start 11185 --seconds 20 --video artifacts/clip.mp4

Rules note (RULES.md): inspecting the video to understand and debug is explicitly allowed. What is
not allowed is turning what you see into labels, hand corrections, or rules and constants tied to
specific frames of this game. This tool only draws; it never writes a submission.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from . import features, io
from games import half_video_blocks

HOME_COLOUR = (230, 60, 60)
AWAY_COLOUR = (60, 110, 230)
BALL_COLOUR = (255, 220, 40)
LABEL_COLOUR = {"H": HOME_COLOUR, "A": AWAY_COLOUR, "D": (170, 170, 170), "X": (90, 90, 90)}


def load_homographies(path, frames):
    """3x3 matrices for the requested frames, from the CSV (frame, 9 matrix entries, ...)."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8")
    order = np.argsort(table["frame"])
    table = table[order]
    names = ("h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22")
    matrices = np.stack([np.stack([table[name] for name in names], axis=1)[frames]])
    return matrices[0]


def project(matrix, x, y):
    """Normalised pitch coordinates -> pixel coordinates under a 3x3 homography."""
    denominator = matrix[6] * x + matrix[7] * y + matrix[8]
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return None
    return ((matrix[0] * x + matrix[1] * y + matrix[2]) / denominator,
            (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator)


def extract_frames(video, start, count, step, width):
    """Decode `count` frames starting at `start`, sampled every `step` frames, scaled to `width`."""
    folder = tempfile.mkdtemp(prefix="mtah_frames_")
    seconds = max((count - 1) * step, 1) / float(io.FPS) + 0.2
    pattern = os.path.join(folder, "f%05d.png")
    command = ["ffmpeg", "-v", "error", "-ss", "{:.3f}".format(start / float(io.FPS)),
               "-t", "{:.3f}".format(seconds), "-i", video,
               "-vf", "select='not(mod(n\\,{}))',scale={}:-2".format(step, width),
               "-vsync", "0", pattern]
    subprocess.run(command, check=True)
    paths = sorted(os.path.join(folder, name) for name in os.listdir(folder))
    return folder, paths


def annotate(image, frame, matrix, slots, ball, labels, team_of_slot, evidence, scale):
    """Draw players, the ball and a text panel on one frame. ``team_of_slot`` is 0=home, 1=away."""
    draw = ImageDraw.Draw(image, "RGBA")
    for index in range(slots.shape[0]):
        x, y = slots[index, 0], slots[index, 1]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        point = project(matrix, x, y)
        if point is None or not all(np.isfinite(point)):
            continue
        px, py = point[0] * scale, point[1] * scale
        if not (0 <= px < image.width and 0 <= py < image.height):
            continue
        colour = HOME_COLOUR if team_of_slot[index] == 0 else AWAY_COLOUR
        radius = 7
        draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                     outline=colour, width=2)

    if np.isfinite(ball[0]):
        point = project(matrix, ball[0], ball[1])
        if point and all(np.isfinite(point)):
            px, py = point[0] * scale, point[1] * scale
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=BALL_COLOUR, outline=(0, 0, 0))

    label = labels[frame]
    panel = "frame {}  t={:.1f}s   label {}".format(frame, frame / float(io.FPS), label)
    detail = evidence
    draw.rectangle((8, 8, 8 + 430, 8 + 46), fill=(0, 0, 0, 150))
    draw.text((16, 14), panel, fill=LABEL_COLOUR.get(label, (255, 255, 255)))
    draw.text((16, 34), detail, fill=(230, 230, 230))
    return image


def evidence_line(frame, X, column):
    gap = max(X[frame, column["ball_gap_prev"]], X[frame, column["ball_gap_next"]])
    parts = [
        "ball {}".format("seen" if X[frame, column["ball_present"]] > 0.5 else "MISSING"),
        "gap {:.1f}s".format(gap / io.FPS),
        "margin {:+.2f}".format(X[frame, column["margin_top3"]]),
        "speed {:.1f}".format(X[frame, column["player_speed_mean"]]),
    ]
    if X[frame, column["in_set_piece"]] > 0.5:
        parts.append("SET PIECE")
    if X[frame, column["ball_outside_pitch"]] > 0.5:
        parts.append("OUT OF PLAY")
    return "  ".join(parts)


def contact_sheet(images, labels_text, columns=4):
    """Tile annotated frames into one image, captioned."""
    if not images:
        raise SystemExit("no frames decoded")
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (20, 20, 20))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    return sheet


def main(argv=None):
    parser = argparse.ArgumentParser(description="render annotated frames for visual verification")
    parser.add_argument("--start", type=int, required=True, help="first video frame")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--every", type=int, default=25, help="sample every N frames")
    parser.add_argument("--width", type=int, default=640, help="frame width in the sheet")
    parser.add_argument("--out", default=os.path.join(features.ARTIFACTS, "verify.png"))
    parser.add_argument("--video", default=None, help="also write an annotated mp4 here")
    parser.add_argument("--submission", default=None, help="frame,label CSV (default: decode)")
    args = parser.parse_args(argv)

    paths = io.asset_paths()
    X, names = features.load()
    column = {name: j for j, name in enumerate(names)}

    if args.submission:
        table = np.genfromtxt(args.submission, delimiter=",", names=True, dtype=None, encoding="utf8")
        labels = np.asarray(table["label"], dtype="<U1")
    else:
        from . import decode
        labels = decode.decode(X, names)

    assets = io.load_all()
    slots_all, ball_all, teams = assets["slots"], assets["ball"], assets["slot_teams"]

    count = int(args.seconds * io.FPS // args.every) + 1
    frames = [args.start + i * args.every for i in range(count)]
    frames = [f for f in frames if 0 <= f < io.N_FRAMES]

    folder, paths_png = extract_frames(paths["video"], args.start, count, args.every, args.width)
    try:
        matrices = load_homographies(paths["homography"], frames)
        images = []
        for frame, png, matrix in zip(frames, paths_png, matrices):
            image = Image.open(png).convert("RGB")
            scale = image.width / float(1920)
            order = np.argsort(teams, kind="stable")      # home first, so colours line up
            image = annotate(image, frame, matrix, slots_all[frame][order], ball_all[frame],
                             labels, teams[order], evidence_line(frame, X, column), scale)
            images.append(image)
        sheet = contact_sheet(images, labels)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        sheet.save(args.out)
        print("wrote {} ({} frames, {}x{})".format(os.path.relpath(args.out, io.ROOT),
                                                   len(images), sheet.width, sheet.height))

        if args.video:
            command = ["ffmpeg", "-v", "error", "-y", "-framerate",
                       "{:g}".format(io.FPS / args.every), "-i",
                       os.path.join(folder, "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       args.video]
            subprocess.run(command, check=True)
            print("wrote {}".format(os.path.relpath(args.video, io.ROOT)))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
