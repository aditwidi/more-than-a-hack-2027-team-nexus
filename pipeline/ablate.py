"""Ablation harness: what does each prior cost?

Every decoder variant is scored twice:

* **label-free** -- possession count, mean duration, dead share, transitions per minute, and
  agreement with restarts. These are the numbers the defaults are chosen on, because they can be
  measured on all 154 minutes that carry no labels, and because they are what the ranking metric is
  built from.
* **public 5 minutes** -- reported for *correctness only*. `EVALUATION.md` demonstrates that
  selecting a hyperparameter on this window picks the worse full-match variant even for a
  one-parameter baseline, and `RULES.md` forbids using it as a tuning set. It is printed so that a
  wrong turn (e.g. a decoder that reports 900 possessions) is visible, not so that the best public
  number can be picked.

    python -m pipeline.ablate --write     # writes evaluation/results/abl_*.csv
    python evaluation/run_evaluation.py   # scores them all at once
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from . import decode, diag, emissions, features, io
from games import half_video_blocks

# name -> (overrides, use_cleanup).
#
# Ordered so each block isolates one decision: the ablation table in the technical document is this
# dictionary. The last entry is the configuration the submission uses.
VARIANTS = {
    "abl_argmax": ({"switch_penalty": 0.0, "switch_penalty_dead": 0.0}, False),
    "abl_no_context": ({"possession_weights": (1.0, 0.0, 0.0)}, False),
    "abl_no_gate": ({"possession_gate_factor": 1.0}, False),
    "abl_no_confidence": ({"possession_confidence_s": 1e6, "possession_gate_factor": 1.0}, False),
    "abl_switch2": ({"switch_penalty": 2.0}, False),
    "abl_switch4": ({"switch_penalty": 4.0}, False),
    "abl_switch6": ({"switch_penalty": 6.0}, False),
    "abl_switch10": ({"switch_penalty": 10.0}, False),
    "abl_switch6_dwell50": ({"min_possession_frames": 50}, True),
    "abl_switch6_dwell75": ({"min_possession_frames": 75}, True),
    "abl_switch6_dead50": ({"min_dead_frames": 50}, True),
    "abl_final": ({}, True),
}


def label_free(labels, X, names):
    halves = {}
    for half, first, last in half_video_blocks("BAR-ATH"):
        stats = diag.structure(labels, first, last)
        halves["half{}".format(half)] = stats
    restarts = diag.restart_agreement(X, names, labels)
    possessions = sum(h["possessions"] for h in halves.values())
    mean_duration = float(np.mean([h["mean_possession_s"] for h in halves.values()]))
    return {
        "possessions": possessions,
        "per_half": [halves["half1"]["possessions"], halves["half2"]["possessions"]],
        "mean_s": round(mean_duration, 2),
        "short_lt2s": sum(h["short_possessions_lt_2s"] for h in halves.values()),
        "dead_share": round(float(np.mean([h["dead_share"] for h in halves.values()])), 3),
        "home_share": round(float(np.mean([h["home_share_alive"] for h in halves.values()])), 3),
        "trans_per_min": round(float(np.mean([h["transitions_per_min"] for h in halves.values()])), 1),
        "set_piece_dead": restarts["set_piece_frames_predicted_dead"],
        "dead_explained": restarts["dead_frames_explained"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="ablate the decoder priors")
    parser.add_argument("--write", action="store_true",
                        help="write each variant to evaluation/results/ for scoring")
    parser.add_argument("--out-dir", default=os.path.join(io.ROOT, "evaluation", "results"))
    parser.add_argument("--features", default=None)
    args = parser.parse_args(argv)

    if args.features:
        data = np.load(args.features)
        X, names = data["X"], [str(n) for n in data["names"]]
    else:
        X, names = features.load()

    rows = []
    for name, (params, use_cleanup) in VARIANTS.items():
        labels = decode.decode(X, names, params=params, use_cleanup=use_cleanup)
        stats = label_free(labels, X, names)
        stats["variant"] = name
        rows.append(stats)
        if args.write:
            decode.write_submission(os.path.join(args.out_dir, "{}.csv".format(name)), labels)

    header = ["variant", "possessions", "per_half", "mean_s", "short_lt2s", "dead_share",
              "home_share", "trans_per_min", "set_piece_dead", "dead_explained"]
    print("\nLabel-free measurements. Choose on these (band: ~100-200 possessions per half, mean "
          "duration 8-15 s).")
    print("  " + "".join("{:<30}".format(h) if h == "variant" else "{:<14}".format(h)
                         for h in header))
    for row in rows:
        print("  " + "{:<30}{:<14}{:<14}{:<14}{:<14}{:<14}{:<14}{:<14}{:<14}{:<14}".format(
            row["variant"], row["possessions"], str(row["per_half"]), row["mean_s"],
            row["short_lt2s"], row["dead_share"], row["home_share"], row["trans_per_min"],
            row["set_piece_dead"], row["dead_explained"]))

    if args.write:
        print("\nwrote {} variant files to {}".format(len(rows), os.path.relpath(args.out_dir, io.ROOT)))
        print("score them: python evaluation/run_evaluation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
