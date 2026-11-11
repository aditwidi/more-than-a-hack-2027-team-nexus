"""Label-free diagnostics for the possession pipeline.

The competition gives 5 minutes of ground truth and says, loudly, not to tune on it
([EVALUATION.md](../../EVALUATION.md)). That leaves a problem: how do you know whether a change is
an improvement on the other 154 minutes? This module answers it with measurements that need no
labels at all -- properties the data itself guarantees:

* **Ball coverage and gap lengths.** If a change makes the ball absent on more frames than the
  source does, it is broken, not improved.
* **Dead-ball agreement with restarts.** Every set-piece interval in Smart Tagging begins with the
  ball out of play. A dead-ball detector that never fires inside set pieces is not a dead-ball
  detector. (Smart Tagging is inferred, so this is a proxy, not a score.)
* **Possession structure.** Count, mean duration, home share, transitions per minute. The ranking
  metric is built on these, and a system reporting 130 possessions in a 5-minute window is wrong
  regardless of its frame accuracy.
* **Control row.** A nearest-player control submission is written to `evaluation/results/` so the
  official scorer always has something to compare against on the public window.

Usage:
    python -m pipeline.features      # once, ~4 s + 3 s parse, writes artifacts/features.npz
    python -m pipeline.diag          # prints the report, writes artifacts/diag.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import features, io
from games import half_video_blocks

LABELS = ("H", "A", "D")
PUBLIC_META = os.path.join(io.ROOT, "evaluation", "gt",
                          "BAR-ATH_possession_gt_public_meta.json")


# ---------------------------------------------------------------------------------------------
# Runs and possessions
# ---------------------------------------------------------------------------------------------
def runs(labels):
    """(start, end, label) for every maximal run, end inclusive."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    edges = np.concatenate(([0], change, [len(labels)]))
    return [(int(edges[i]), int(edges[i + 1] - 1), str(labels[edges[i]]))
            for i in range(len(edges) - 1)]


def possessions(labels):
    """Maximal runs of H or A -- the definition the ranking metric is built on.

    Dead-ball stretches are stoppages, not possessions, so they break a run without creating one:
    `H H D D H H` is two possessions, not one (CHALLENGE.md, "What is already settled").
    """
    spells = []
    for start, end, label in runs(labels):
        if label in ("H", "A"):
            spells.append({"start": start, "end": end, "team": label,
                           "length": end - start + 1})
    return spells


def structure(labels, first, last):
    """Possession-structure statistics over one half, in the units a coach would use."""
    window = labels[first:last + 1]
    spells = possessions(window)
    lengths = np.array([s["length"] for s in spells], dtype=float)
    alive = np.isin(window, ("H", "A"))
    transitions = 0
    for a, b in zip(window[:-1], window[1:]):
        if a in ("H", "A", "D") and b in ("H", "A", "D") and a != b:
            transitions += 1
    duration_s = (last - first + 1) / float(io.FPS)
    home_frames = int(np.sum(window == "H"))
    away_frames = int(np.sum(window == "A"))
    return {
        "frames": int(window.size),
        "possessions": len(spells),
        "possessions_per_min": round(len(spells) / max(duration_s / 60.0, 1e-9), 2),
        "mean_possession_s": round(float(lengths.mean()) / io.FPS, 2) if lengths.size else 0.0,
        "median_possession_s": round(float(np.median(lengths)) / io.FPS, 2) if lengths.size else 0.0,
        "p10_possession_s": round(float(np.percentile(lengths, 10)) / io.FPS, 2) if lengths.size else 0.0,
        "short_possessions_lt_2s": int(np.sum(lengths < 2 * io.FPS)),
        "dead_share": round(float(np.mean(window == "D")), 4),
        "home_share_alive": round(home_frames / max(home_frames + away_frames, 1), 4),
        "transitions": int(transitions),
        "transitions_per_min": round(transitions / max(duration_s / 60.0, 1e-9), 2),
        "alive_share": round(float(alive.mean()), 4),
    }


# ---------------------------------------------------------------------------------------------
# Control labels
# ---------------------------------------------------------------------------------------------
def control_labels(X, names):
    """Nearest-team labels with `D` wherever the ball is not visible.

    This is the published baseline's rule (`starter/baseline.py`) expressed on the feature matrix,
    so the board has a control row that is cheap to regenerate and cannot drift from the features.
    It is a control, not the submission: no smoothing, no dead-ball evidence, no structure.
    """
    column = {name: j for j, name in enumerate(names)}
    present = X[:, column["ball_present"]] > 0.5
    margin = X[:, column["score_margin"]]

    labels = np.full(X.shape[0], "D", dtype="<U1")
    labels[present] = np.where(margin[present] >= 0, "H", "A")
    return labels


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def ball_report(X, names):
    column = {name: j for j, name in enumerate(names)}
    present = X[:, column["ball_present"]] > 0.5
    gap = np.maximum(X[:, column["ball_gap_prev"]], X[:, column["ball_gap_next"]])
    graded = X[:, column["is_graded"]] > 0.5

    report = {
        "ball_present_all": round(float(present.mean()), 4),
        "ball_present_graded": round(float(present[graded].mean()), 4),
        "absent_runs": 0, "absent_run_p50_frames": 0, "absent_run_p90_frames": 0,
        "absent_run_p99_frames": 0, "absent_run_max_frames": 0,
        "graded_frames_in_gap_over_1s": 0,
        "graded_frames_in_gap_over_3s": 0,
    }
    gaps = [end - start + 1 for start, end, label in runs(np.where(present[graded], "P", "-"))
            if label == "-"]
    if gaps:
        gaps = np.array(gaps)
        report.update({
            "absent_runs": int(gaps.size),
            "absent_run_p50_frames": int(np.percentile(gaps, 50)),
            "absent_run_p90_frames": int(np.percentile(gaps, 90)),
            "absent_run_p99_frames": int(np.percentile(gaps, 99)),
            "absent_run_max_frames": int(gaps.max()),
        })
    report["graded_frames_in_gap_over_1s"] = int(np.sum(graded & (gap > io.FPS)))
    report["graded_frames_in_gap_over_3s"] = int(np.sum(graded & (gap > 3 * io.FPS)))
    return report


def tag_report(X, names, data_dir=None):
    column = {name: j for j, name in enumerate(names)}
    set_piece = X[:, column["in_set_piece"]] > 0.5
    graded = X[:, column["is_graded"]] > 0.5
    team = X[:, column["tag_team"]]

    # Instance counts come from the XML rather than the feature matrix: they are a property of the
    # file, and they are the numbers DATA.md publishes, so they are worth showing side by side.
    try:
        tags = io.load_smart_tagging(io.asset_paths(data_dir=data_dir)["smart_tagging"])
        instances = dict(sorted(tags["code_instances"].items(), key=lambda kv: -kv[1]))
        groups = tags["meta"]
    except Exception as error:            # the data bundle is optional for the report
        instances, groups = {}, {"error": str(error)}

    report = {
        "set_piece_frames": int(np.sum(set_piece)),
        "set_piece_frames_graded": int(np.sum(set_piece & graded)),
        "set_piece_share_graded": round(float(np.mean(set_piece[graded])), 4),
        "tag_team_frames_home": int(np.sum(team > 0)),
        "tag_team_frames_away": int(np.sum(team < 0)),
        "tag_team_frames_unknown": int(np.sum(team == 0)),
        "code_instances": instances,
        "tag_groups": groups,
    }
    return report


def restart_agreement(X, names, labels):
    """Dead-ball proxy: does `D` line up with restarts and long absences, and not with play?

    Two directions, neither of which needs a label:

    * **recall proxy** -- of the frames inside a set-piece interval, what share are predicted `D`?
      A set piece starts with the ball out of play, so a dead-ball detector should fire there. It
      will not cover the whole interval (set pieces contain live play too), so this is a floor.
    * **precision proxy** -- of the predicted `D` frames, what share sit inside a set piece, inside
      a ball absence longer than a second, or in the dead frame block before the second-half
      kickoff? `D` with none of those three explanations is a guess.
    """
    column = {name: j for j, name in enumerate(names)}
    graded = X[:, column["is_graded"]] > 0.5
    set_piece = X[:, column["in_set_piece"]] > 0.5
    gap = np.maximum(X[:, column["ball_gap_prev"]], X[:, column["ball_gap_next"]])
    long_gap = gap > io.FPS
    dead = labels == "D"

    _, second_first, second_last = half_video_blocks("BAR-ATH")[1]
    pre_kickoff = np.zeros_like(graded)
    pre_kickoff[second_first:second_first + 2934] = True      # FAQ: all D, no kickoff yet

    explained = set_piece | long_gap | pre_kickoff
    return {
        "set_piece_frames_predicted_dead": round(float(np.mean(dead[set_piece])) if set_piece.any() else 0.0, 4),
        "dead_frames_explained": round(float(np.mean(explained[dead])) if dead.any() else 0.0, 4),
        "dead_frames_on_graded": int(np.sum(dead & graded)),
        "dead_share_graded": round(float(np.mean(dead[graded])), 4),
    }


def public_reference():
    """Published statistics of the 5-minute public window -- printed as a sanity reference only.

    These are *not* a tuning target: EVALUATION.md shows that choosing a hyperparameter against
    this window picks the worse full-match variant even for a one-parameter baseline.
    """
    if not os.path.isfile(PUBLIC_META):
        return None
    with open(PUBLIC_META) as handle:
        meta = json.load(handle)
    return {
        "n_possessions": meta["n_possessions"],
        "mean_possession_s": meta["mean_possession_duration_s"],
        "label_counts": meta["label_counts_in_window"],
        "window": meta["public_window"],
    }


def report(X, names, verbose=True, data_dir=None):
    labels = control_labels(X, names)
    halves = {}
    for half, first, last in half_video_blocks("BAR-ATH"):
        halves["half{}".format(half)] = structure(labels, first, last)

    first, last = half_video_blocks("BAR-ATH")[0][1], half_video_blocks("BAR-ATH")[1][2]
    out = {
        "control": "nearest tracked team to the ball, D where the ball is absent",
        "ball": ball_report(X, names),
        "tags": tag_report(X, names, data_dir=data_dir),
        "restarts": restart_agreement(X, names, labels),
        "structure": halves,
        "structure_full": structure(labels, first, last),
        "public_reference": public_reference(),
    }

    if verbose:
        _print(out)
    return labels, out


def _print(out):
    def line(key, value):
        print("  {:<42} {}".format(key, value))

    print("\n=== ball visibility (source, not a prediction) ===")
    for key, value in out["ball"].items():
        line(key, value)

    print("\n=== Smart Tagging coverage ===")
    for key, value in out["tags"].items():
        if key not in ("code_instances", "tag_groups"):
            line(key, value)
    instances = list(out["tags"]["code_instances"].items())[:6]
    line("code instances (top)", ", ".join("{}={}".format(k, v) for k, v in instances))
    groups = out["tags"]["tag_groups"]
    for group in ("Team", "Half", "Type"):
        if group in groups:
            line("{} tag".format(group), groups[group])

    print("\n=== control labels: structure (label-free) ===")
    for name, stats in out["structure"].items():
        print("  {}".format(name))
        for key, value in stats.items():
            print("    {:<38} {}".format(key, value))

    print("\n=== control labels: dead-ball proxies ===")
    for key, value in out["restarts"].items():
        line(key, value)

    print("\n=== public 5-minute window, published statistics (reference only, not a target) ===")
    reference = out["public_reference"]
    if reference:
        line("possessions", reference["n_possessions"])
        line("mean possession (s)", reference["mean_possession_s"])
        line("label counts", reference["label_counts"])
    else:
        line("unavailable", "evaluation/gt/..._public_meta.json not found")


def main(argv=None):
    parser = argparse.ArgumentParser(description="label-free diagnostics for the pipeline")
    parser.add_argument("--features", default=None, help="features npz (default: artifacts/)")
    parser.add_argument("--data-dir", default=None, help="match bundle (default: $MTAH_DATA_DIR)")
    parser.add_argument("--submit", action="store_true",
                        help="also write the control submission to evaluation/results/")
    parser.add_argument("--out", default=os.path.join(features.ARTIFACTS, "diag.json"))
    args = parser.parse_args(argv)

    if args.features:
        data = np.load(args.features)
        X, names = data["X"], [str(n) for n in data["names"]]
    else:
        X, names = features.load()

    labels, out = report(X, names, data_dir=args.data_dir)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
    print("\nwrote {}".format(os.path.relpath(args.out, io.ROOT)))

    if args.submit:
        path = write_submission(
            os.path.join(io.ROOT, "evaluation", "results", "control_nearest.csv"), labels)
        print("wrote {} ({:,} rows) -- now: python evaluation/run_evaluation.py"
              .format(os.path.relpath(path, io.ROOT), len(labels)))


if __name__ == "__main__":
    main()
