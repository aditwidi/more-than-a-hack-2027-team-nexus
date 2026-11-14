"""Render annotated frames: project the tracking onto the video and burn in our own labels.

This is the visual half of verification (the quantitative half is `pipeline/verify.py`). It answers
questions the numbers cannot: *does the predicted label match what is on the pitch in this frame*,
and *why did the model think so*.

For each frame it takes the homography row for that frame, projects every tracked slot (normalised
pitch coordinates, the same convention the ATD uses) into image pixels, draws players by team and
the ball, and prints the label plus the evidence the emission used. With `--gt` it also prints the
public ground-truth label and flags disagreement in red — the 5 published minutes exist exactly for
this kind of check.

    # the 5 published minutes, with ground truth side by side (frames 11,185-18,684)
    python -m pipeline.render --start 11185 --end 18684 --gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
        --video artifacts/public_window.mp4 --fps 25 --width 960

    # a contact sheet of one moment
    python -m pipeline.render --start 45900 --seconds 4 --out artifacts/goal.png

    # the whole match, streamed (no disk staging)
    python -m pipeline.render --full --video artifacts/full.mp4 --fps 5 --width 960

Decoding and encoding are both pipes: ffmpeg -> raw RGB -> annotate in memory -> ffmpeg. Nothing is
staged on disk, so memory stays flat regardless of length.

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
import pandas as pd
from PIL import Image, ImageDraw

from . import features, io
from games import half_video_blocks

HOME_COLOUR = (230, 60, 60)
AWAY_COLOUR = (60, 110, 230)
BALL_COLOUR = (255, 220, 40)
LABEL_COLOUR = {"H": (255, 90, 90), "A": (110, 150, 255), "D": (190, 190, 190), "X": (120, 120, 120)}
DISAGREE_COLOUR = (255, 40, 40)


def load_homography_matrices(path):
    """(n_frames, 9) homography rows, in frame order. Loaded once -- never inside a frame loop."""
    table = pd.read_csv(path).sort_values("frame")
    columns = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
    return table[columns].to_numpy(dtype=np.float64)


def project(matrix, x, y):
    """Normalised pitch coordinates -> pixel coordinates under a 3x3 homography."""
    denominator = matrix[6] * x + matrix[7] * y + matrix[8]
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return None
    return ((matrix[0] * x + matrix[1] * y + matrix[2]) / denominator,
            (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator)


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


def annotate(image, frame, matrix, slots, ball, team_of_slot, text, scale=1.0, gt=None):
    """Draw players, the ball, the pitch outline and a text panel.

    ``scale`` converts projected pixels (always computed at 1920x1080) to the rendered image size --
    forgetting it puts every circle at the wrong offset, which looks exactly like a broken
    homography. The pitch rectangle is drawn from the same matrix, so alignment can be checked
    against the touchlines in the image itself.
    """
    draw = ImageDraw.Draw(image, "RGBA")

    outline = []
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        point = project(matrix, x, y)
        if point and all(np.isfinite(point)):
            outline.append((point[0] * scale, point[1] * scale))
    if len(outline) == 4:
        draw.line(outline + [outline[0]], fill=(255, 255, 255), width=2)

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
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=colour, width=2)

    if np.isfinite(ball[0]):
        point = project(matrix, ball[0], ball[1])
        if point and all(np.isfinite(point)):
            px, py = point[0] * scale, point[1] * scale
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=BALL_COLOUR, outline=(0, 0, 0))

    disagree = gt is not None and gt in ("H", "A", "D") and gt != text.split()[0]
    draw.rectangle((8, 8, 8 + 470, 8 + 48), fill=(0, 0, 0, 165))
    if disagree:
        draw.rectangle((8, 8, 8 + 470, 8 + 48), outline=DISAGREE_COLOUR, width=3)
    draw.text((16, 13), text, fill=(255, 255, 255))
    detail = "gt {}".format(gt) if gt is not None else ""
    draw.text((16, 32), "{}   {}".format(evidence_line_cached, detail), fill=(225, 225, 225))
    return image


# `annotate` is hot (one call per frame), so the evidence string is passed in a module-level slot
# rather than recomputed: it is built by the caller, which already has the feature matrix row.
evidence_line_cached = ""


def render_stream(video_path, matrices, slots_all, ball_all, teams, labels, texts, gt_labels,
                  start, end, out_path, width=960, fps=5.0, progress=500):
    """Render [start, end] frame by frame, streamed: ffmpeg -> memory -> ffmpeg.

    The whole match is 199,319 frames; staging scaled PNGs first would need ~100 GB, so both ends
    are pipes. `fps` is the output rate, not the source rate (25 fps, two hours of video).
    """
    height = int(round(width * 1080 / 1920 / 2)) * 2
    frame_bytes = width * height * 3
    step = io.FPS / fps

    seconds = (end - start + 1) / float(io.FPS)
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", "{:.3f}".format(start / float(io.FPS)),
         "-i", video_path, "-t", "{:.3f}".format(seconds),
         "-vf", "fps={:g},scale={}:{}".format(fps, width, height),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "{}x{}".format(width, height), "-r", "{:g}".format(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         out_path],
        stdin=subprocess.PIPE)

    global evidence_line_cached
    order = np.argsort(teams, kind="stable")
    index = 0
    try:
        while True:
            buffer = decoder.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            frame = min(int(round(start + index * step)), io.N_FRAMES - 1)
            image = Image.frombytes("RGB", (width, height), buffer)

            evidence_line_cached = texts[frame]
            text = "frame {}  t={:.1f}s   label {}".format(
                frame, frame / float(io.FPS), labels[frame])
            gt = gt_labels[frame] if gt_labels is not None else None
            annotate(image, frame, matrices[frame], slots_all[frame][order], ball_all[frame],
                     teams[order], text, scale=width / 1920.0, gt=gt)
            encoder.stdin.write(image.tobytes())

            index += 1
            if progress and index % progress == 0:
                total = max((end - start) * fps / io.FPS, 1)
                print("  {:,} frames ({:.0f}%)".format(index, 100.0 * index / total),
                      flush=True)
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()
    return index


def extract_frames(video, start, count, step, width):
    """Decode `count` frames starting at `start`, sampled every `step` frames, scaled to `width`."""
    folder = tempfile.mkdtemp(prefix="mtah_frames_")
    seconds = max((count - 1) * step, 1) / float(io.FPS) + 0.2
    pattern = os.path.join(folder, "f%05d.png")
    subprocess.run(["ffmpeg", "-v", "error", "-ss", "{:.3f}".format(start / float(io.FPS)),
                    "-t", "{:.3f}".format(seconds), "-i", video,
                    "-vf", "select='not(mod(n\\,{}))',scale={}:-2".format(step, width),
                    "-vsync", "0", pattern], check=True)
    return folder, sorted(os.path.join(folder, name) for name in os.listdir(folder))


def contact_sheet(images, columns=4):
    if not images:
        raise SystemExit("no frames decoded")
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (20, 20, 20))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    return sheet


def load_labels(path):
    table = pd.read_csv(path)
    labels = np.asarray(table["label"].astype(str).str.strip().str.upper(), dtype="<U1")
    if len(labels) != io.N_FRAMES:
        raise SystemExit("{}: {} rows, expected {}".format(path, len(labels), io.N_FRAMES))
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description="render annotated frames for visual verification")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="last frame (inclusive)")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--full", action="store_true", help="the whole match")
    parser.add_argument("--every", type=int, default=25, help="frame step for contact sheets")
    parser.add_argument("--fps", type=float, default=0.0, help="output fps for video mode")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--out", default=os.path.join(features.ARTIFACTS, "verify.png"))
    parser.add_argument("--video", default=None, help="write an annotated mp4 instead of a sheet")
    parser.add_argument("--submission", default=None, help="frame,label CSV (default: decode)")
    parser.add_argument("--gt", default=None, help="ground-truth CSV to show side by side")
    args = parser.parse_args(argv)

    paths = io.asset_paths()
    X, names = features.load()
    column = {name: j for j, name in enumerate(names)}

    if args.submission:
        labels = load_labels(args.submission)
    else:
        from . import decode
        labels = decode.decode(X, names)

    gt_labels = None
    if args.gt:
        gt_labels = load_labels(args.gt)
        graded = gt_labels != "X"
        if graded.any():
            print("ground truth covers frames {:,}..{:,} ({:,} graded)".format(
                int(np.flatnonzero(graded)[0]), int(np.flatnonzero(graded)[-1]),
                int(graded.sum())))

    assets = io.load_all()
    slots_all, ball_all, teams = assets["slots"], assets["ball"], assets["slot_teams"]
    texts = [evidence_line(f, X, column) for f in range(0, io.N_FRAMES, 1)] if False else None
    matrices = load_homography_matrices(paths["homography"])
    print("loaded {:,} homographies".format(len(matrices)))

    if args.video:
        end = io.N_FRAMES - 1 if args.full else (
            args.end if args.end is not None else
            min(io.N_FRAMES - 1, args.start + int(args.seconds * io.FPS)))
        fps = args.fps if args.fps > 0 else (5.0 if args.full else io.FPS / args.every)
        print("rendering frames {:,}..{:,} at {:g} fps, {} px -> {}".format(
            args.start, end, fps, args.width, os.path.relpath(args.video, io.ROOT)))
        rendered = render_stream(
            paths["video"], matrices, slots_all, ball_all, teams, labels,
            EvidenceCache(X, column), gt_labels, args.start, end, args.video,
            width=args.width, fps=fps)
        print("wrote {} ({:,} frames)".format(os.path.relpath(args.video, io.ROOT), rendered))
        return 0

    count = int(args.seconds * io.FPS // args.every) + 1
    frames = [args.start + i * args.every for i in range(count)]
    frames = [f for f in frames if 0 <= f < io.N_FRAMES]

    folder, pngs = extract_frames(paths["video"], args.start, count, args.every, args.width)
    try:
        order = np.argsort(teams, kind="stable")
        images = []
        for frame, png in zip(frames, pngs):
            image = Image.open(png).convert("RGB")
            global evidence_line_cached
            evidence_line_cached = evidence_line(frame, X, column)
            text = "frame {}  t={:.1f}s   label {}".format(
                frame, frame / float(io.FPS), labels[frame])
            gt = gt_labels[frame] if gt_labels is not None else None
            if gt is not None:
                text += "  (gt {})".format(gt)
            annotate(image, frame, matrices[frame], slots_all[frame][order], ball_all[frame],
                     teams[order], text, scale=image.width / 1920.0, gt=gt)
            images.append(image)
        sheet = contact_sheet(images)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        sheet.save(args.out)
        print("wrote {} ({} frames, {}x{})".format(os.path.relpath(args.out, io.ROOT),
                                                   len(images), sheet.width, sheet.height))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return 0


class EvidenceCache:
    """Lazily-formatted evidence strings, one per frame (cheap: built on access, not upfront)."""

    def __init__(self, X, column):
        self.X, self.column = X, column

    def __getitem__(self, frame):
        return evidence_line(frame, self.X, self.column)


if __name__ == "__main__":
    raise SystemExit(main())
