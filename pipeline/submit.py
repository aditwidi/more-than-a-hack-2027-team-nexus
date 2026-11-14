"""Build the submission file from the features, in one command.

    python -m pipeline.submit                          # artifacts/submission.csv
    python -m pipeline.submit --out nexus.csv --score   # write it and score it

The decoder is `pipeline/decode.py`; the priors it uses are in `pipeline/emissions.py` and are
documented there. Nothing in this path reads the ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from . import decode, diag, features, io
from games import half_video_blocks


def build(params=None, use_cleanup=True, features_path=None):
    if features_path:
        data = np.load(features_path)
        X, names = data["X"], [str(n) for n in data["names"]]
    else:
        X, names = features.load()
    labels = decode.decode(X, names, params=params, use_cleanup=use_cleanup)
    return labels, X, names


def summarise(labels):
    """Possession structure of a finished label sequence, per half."""
    out = {}
    for half, first, last in half_video_blocks("BAR-ATH"):
        out["half{}".format(half)] = diag.structure(labels, first, last)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="write the possession submission")
    parser.add_argument("--out", default=os.path.join(features.ARTIFACTS, "submission.csv"))
    parser.add_argument("--params", default=None,
                        help="JSON string of emission/decoder overrides")
    parser.add_argument("--no-cleanup", action="store_true", help="skip structural cleanup")
    parser.add_argument("--features", default=None, help="features npz (default: artifacts/)")
    parser.add_argument("--score", action="store_true",
                        help="also validate and score against the public window")
    args = parser.parse_args(argv)

    params = json.loads(args.params) if args.params else None
    labels, X, names = build(params, not args.no_cleanup, args.features)

    path = decode.write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    print("wrote {} ({:,} rows)".format(os.path.relpath(path, io.ROOT), len(labels)))
    print("  labels  " + "  ".join("{} {:>6,}".format(k, v) for k, v in counts.items()))

    for half, stats in summarise(labels).items():
        print("  {}  possessions {:>4}  mean {:>5.2f}s  dead {:>5.1%}  home share {:.2f}  "
              "transitions/min {:>5.1f}"
              .format(half, stats["possessions"], stats["mean_possession_s"],
                      stats["dead_share"], stats["home_share_alive"],
                      stats["transitions_per_min"]))

    if args.score:
        for command in (["python", "evaluation/validate_submission.py", path],
                        ["python", "evaluation/run_evaluation.py"]):
            print("\n$ " + " ".join(command))
            subprocess.run(command, cwd=io.ROOT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
