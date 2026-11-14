"""Combine the rule-based emission with a learned one, and measure whether it helps.

Why this shape
--------------
There are no labels for this match, so a supervised model has nothing to learn from except our own
pipeline's output (pseudo-labels) — the "self-training" setting. The literature is clear about what
that can and cannot buy: a student model can smooth and re-weight what the teacher already encodes,
and can pick up interactions the hand-set log-odds miss, but it cannot add information the teacher
never had, and it propagates the teacher's errors. So the honest way to use it is as a **combination**
(hybrid), not a replacement: mix the two probability distributions, decode with the same structured
decoder, and keep the mixture only if it survives measurement.

Validation without labels
-------------------------
Two things are measured, and neither reads the private truth:

* **Out-of-fold agreement with the teacher.** The match is split by half, the model is trained on one
  half and predicts the other. High agreement means the features generalise across halves; it does
  *not* mean the labels are right, and it is reported with that caveat.
* **Label-free structure** of the decoded result (possessions, mean duration, dead share), which must
  stay inside the football priors documented in `docs/references.md`.

    python -m pipeline.learn --alpha 0.5 --write evaluation/results/learned.csv
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import decode, diag, emissions, features, io
from games import half_video_blocks

def dataset(features_path=None):
    if features_path:
        data = np.load(features_path)
        X, names = data["X"], [str(n) for n in data["names"]]
    else:
        X, names = features.load()
    return X, names


def teacher_labels(X, names):
    """Our own pipeline's output, used as the training signal.

    This is algorithmic output, not hand annotation: nothing here was decided by a person, which is
    what RULES.md permits ("pseudo-labels produced by an algorithm").
    """
    return decode.decode(X, names)


def out_of_fold_probabilities(X, y, folds, seed=0):
    """Train on one fold, predict the other, for each fold. Returns (n_frames, 3) probabilities.

    Training and inference never see the same half, so the agreement figure is not memorisation.
    """
    proba = np.zeros((X.shape[0], 3), dtype=np.float32)
    for name, train, test in folds:
        train_mask = np.zeros(X.shape[0], dtype=bool)
        test_mask = np.zeros(X.shape[0], dtype=bool)
        train_mask[train[0]:train[1] + 1] = True
        test_mask[test[0]:test[1] + 1] = True

        model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, max_depth=6, early_stopping=False,
            random_state=seed, verbose=0)
        model.fit(X[train_mask], y[train_mask])
        proba[test_mask] = model.predict_proba(X[test_mask])
        print("  fold: {} -- train {:6d} frames, predict {:6d}"
              .format(name, int(train_mask.sum()), int(test_mask.sum())))
    return proba


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def mix(rule_logits, learned_proba, alpha):
    """Probability mixture of the two emissions. alpha = 0 is rules only, 1 is the learned model."""
    return (1.0 - alpha) * softmax(rule_logits.astype(np.float64)) + alpha * learned_proba


def main(argv=None):
    parser = argparse.ArgumentParser(description="combine rule and learned emissions")
    parser.add_argument("--features", default=None)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--write", default=None, help="write the alpha=best file here")
    parser.add_argument("--out", default=os.path.join(features.ARTIFACTS, "learned.json"))
    args = parser.parse_args(argv)

    X, names = dataset(args.features)
    y_labels = teacher_labels(X, names)
    # Class index by explicit mapping: decode.LABELS is ["H", "A", "D"], which is NOT sorted, so
    # np.searchsorted would collide labels (it mapped H and D to the same index).
    y = np.full(len(y_labels), -1, dtype=np.int64)
    for index, label in enumerate(decode.LABELS):
        y[y_labels == label] = index
    graded = X[:, emissions.column_index(names)["is_graded"]] > 0.5
    assert (y[graded] >= 0).all(), "every graded frame must carry a pseudo-label"

    blocks = half_video_blocks("BAR-ATH")
    first_half, second_half = (blocks[0][1], blocks[0][2]), (blocks[1][1], blocks[1][2])
    folds = [
        ("train 1st half -> predict 2nd", first_half, second_half),
        ("train 2nd half -> predict 1st", second_half, first_half),
    ]

    print("training the learned emission out-of-fold (one half at a time) ...")
    proba = out_of_fold_probabilities(X, y, folds)

    agreement = float(np.mean(proba[graded].argmax(axis=1) == y[graded]))
    print("  out-of-fold agreement with the teacher: {:.4f}".format(agreement))

    rule_logits = emissions.emission_logits(X, names)
    report = {"out_of_fold_agreement": round(agreement, 4), "variants": {}}

    for alpha in args.alpha:
        mixed = mix(rule_logits, proba, alpha)
        logits = np.log(np.clip(mixed, 1e-6, None)).astype(np.float32)
        params = dict(emissions.DEFAULTS)
        states = decode.cleanup(decode.viterbi(logits, params), params)
        labels = decode.LABELS[states].astype("<U1")
        labels[~graded] = "X"

        halves = [diag.structure(labels, first, last) for _h, first, last in blocks]
        stats = {
            "possessions": sum(h["possessions"] for h in halves),
            "mean_s": round(float(np.mean([h["mean_possession_s"] for h in halves])), 2),
            "dead_share": round(float(np.mean([h["dead_share"] for h in halves])), 3),
            "home_share": round(float(np.mean([h["home_share_alive"] for h in halves])), 3),
            "agreement": round(float(np.mean(labels[graded] == y_labels[graded])), 4),
        }
        report["variants"]["alpha={}".format(alpha)] = stats
        print("alpha {:.2f}: possessions {:4d}  mean {:5.2f}s  dead {:.3f}  home {:.3f}  "
              "agreement {:.4f}".format(alpha, stats["possessions"], stats["mean_s"],
                                        stats["dead_share"], stats["home_share"],
                                        stats["agreement"]))

        if args.write and alpha == max(args.alpha):
            path = decode.write_submission(args.write, labels)
            print("  wrote {}".format(os.path.relpath(path, io.ROOT)))

    folder = os.path.dirname(args.out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote {}".format(os.path.relpath(args.out, io.ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
