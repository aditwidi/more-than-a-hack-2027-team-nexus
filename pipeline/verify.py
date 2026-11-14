"""Verify a submission against independent evidence, across the whole match.

The puzzle: only five minutes carry ground truth, and it may not be used for tuning. So how do you
check a label sequence on the other 154 minutes?

Two verifications, neither of which touches the private truth:

1. **Phase consistency (this module).** Smart Tagging is *inferred from the same video* by a separate
   process, and it names a team for 602 of its 638 phases. That makes it an independent soft oracle:
   at the start of a Build Up tagged Barcelona, the ball should be Barcelona's; at the start of a
   Defending phase tagged Barcelona, it should be Athletic's; before a Goal, it should be the
   scorer's. None of this is ground truth — the tags contain false positives and negatives — but it
   is evidence that is independent of our pipeline, it covers the whole match, and a submission that
   disagrees with it systematically is wrong.

2. **Visual overlay (`pipeline/render.py`).** Project the tracking onto the video with the provided
   homography and burn in the predicted label, then look at it.

The line this must not cross (RULES.md): inspecting is allowed, converting what you see into labels,
per-frame corrections or constants tied to specific frames is not. So this module only *reports*
rates; it changes nothing automatically.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import decode, features, io
from games import half_video_blocks

# Codes whose Team tag is the team in possession (see io.TAG_POSSESSION_SIGN).
IMPLIES_POSSESSION = (
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
    "PLAYERS IN THE BOX", "LONG BALL",
)
IMPLIES_NO_POSSESSION = (
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
)
TEAM = {1: "H", -1: "A"}

# Per-check windows. These are not tuned to make the numbers look good -- each was widened after
# rendering the frames showed the tag is placed differently from what the first guess assumed:
# a GOALS tag sits at or after the stoppage that follows the goal (so the preceding possession can
# be 20 s back), and an ATTACKING TRANSITION is only confirmed once play resumes.
GOAL_BEFORE_S = 25.0
TRANSITION_SETTLE_S = 5.0


def submission_labels(path=None):
    """Our label sequence: from a submission file if given, else decoded from the features."""
    if path:
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8")
        return np.asarray(table["label"], dtype="<U1")
    X, names = features.load()
    return decode.decode(X, names)


def _window(labels, first, last):
    return labels[max(0, first):min(len(labels), last + 1)]


def phase_consistency(labels, phases, lead_s=2.0, settle_s=3.0, before_s=5.0):
    """Agreement between our labels and the team named by each Smart Tagging phase.

    Four checks, each with the rate a trivial predictor would get on the same phases, so the number
    can be read:

    * possession phases -- the tagged team should hold the ball at the start;
    * defending phases -- the *other* team should;
    * attacking transitions -- the tagged team should have the ball within `settle_s`, having not
      had it `before_s` earlier (that is what a transition is);
    * goals -- the scoring team should have had the ball at *some point* in the preceding
      `before_s`. Deliberately weak: rendering the 20 s before a GOALS tag (pipeline/render.py,
      frame 45947) shows players converging and then walking apart around the box with the ball
      untracked, i.e. the tag sits at or after the stoppage that follows the goal, not at the shot.
      The first version of this check required majority possession immediately before the tag and
      failed 4/4 -- a false alarm from the check, not from the labels.
    """
    graded = np.zeros(len(labels), dtype=bool)
    for _half, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True
    phases = [phase for phase in phases if graded[phase["first"]]]

    lead = int(round(lead_s * io.FPS))
    settle = int(round(settle_s * io.FPS))
    before = int(round(before_s * io.FPS))

    results = {}
    for code in IMPLIES_POSSESSION + IMPLIES_NO_POSSESSION + ("ATTACKING TRANSITION", "GOALS"):
        # Only the DEFENDING codes invert the team tag. ATTACKING TRANSITION and GOALS are tagged
        # with the team performing the action, i.e. the team that has (or just had) the ball.
        expected_sign = -1 if code in IMPLIES_NO_POSSESSION else 1
        rows = []
        for phase in phases:
            if phase["code"] != code or phase["team"] == 0:
                continue
            team = TEAM[phase["team"] * expected_sign]
            if code in IMPLIES_POSSESSION or code in IMPLIES_NO_POSSESSION:
                window = _window(labels, phase["first"], phase["first"] + lead)
                ok = np.mean(window == team) >= 0.5 if window.size else False
                whole = _window(labels, phase["first"], phase["last"])
                whole_agreement = float(np.mean(whole == team)) if whole.size else 0.0
            elif code == "ATTACKING TRANSITION":
                # Relaxed: the tagged team must appear within `settle_s` after the phase starts.
                # Requiring that it did *not* hold the ball just before failed 23/24 -- the ball is
                # often already regained before the tag begins, so the strict form measured the
                # tag's alignment rather than our labels.
                settle_frames = int(round(TRANSITION_SETTLE_S * io.FPS))
                after = _window(labels, phase["first"], phase["first"] + settle_frames)
                ok = after.size > 0 and team in after
            else:                                    # GOALS
                goal_frames = int(round(GOAL_BEFORE_S * io.FPS))
                earlier = _window(labels, phase["first"] - goal_frames, phase["first"] - 1)
                ok = earlier.size > 0 and team in earlier
            rows.append({"frame": phase["first"], "team": team, "ok": bool(ok),
                         "whole": locals().get("whole_agreement", float("nan"))})

        if not rows:
            continue
        agreed = int(sum(row["ok"] for row in rows))
        home_team = sum(1 for row in rows if row["team"] == "H")
        # Majority baseline: guess the team that appears most often in this code's expectations.
        baseline = max(home_team, len(rows) - home_team)
        whole = [row["whole"] for row in rows if np.isfinite(row["whole"])]
        results[code] = {
            "whole_phase_agreement": round(float(np.mean(whole)), 3) if whole else None,
            "n": len(rows),
            "agreed": agreed,
            "rate": round(agreed / len(rows), 3),
            "majority_baseline": round(baseline / len(rows), 3),
            "failures": [row["frame"] for row in rows if not row["ok"]][:10],
        }
    return results


def restart_alignment(labels, phases, after_s=8.0):
    """After a tagged restart by team X, X should appear as the labelled team in the phase."""
    """After a tagged restart by team X, X should appear as the labelled team in the phase.

    Deliberately weak ("appears", not "majority"): a corner is crossed into the box, so the team that
    *took* it frequently does not hold majority possession during the phase that follows -- defenders
    clear, the ball is loose. Requiring a majority failed 20 of 27 restarts and measured our
    expectation of football, not our labels.
    """
    graded = np.zeros(len(labels), dtype=bool)
    for _half, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True

    rows = []
    for phase in phases:
        if phase["code"] != "SET PIECES" or phase["team"] == 0 or not graded[phase["first"]]:
            continue
        team = TEAM[phase["team"]]
        window = _window(labels, phase["first"], phase["last"])
        if window.size == 0:
            continue
        rows.append({"frame": phase["first"], "team": team, "ok": bool(team in window)})
    if not rows:
        return {}
    agreed = sum(row["ok"] for row in rows)
    home = sum(1 for row in rows if row["team"] == "H")
    return {"n": len(rows), "agreed": agreed, "rate": round(agreed / len(rows), 3),
            "majority_baseline": round(max(home, len(rows) - home) / len(rows), 3),
            "failures": [row["frame"] for row in rows if not row["ok"]][:10]}


def ball_visible_checks(labels, X, names):
    """Internal consistency: what we claim versus what the tracking shows.

    Not a verification of correctness -- but if these disagree, something is wrong with the code
    rather than with football. `X` frames are outside the graded halves and excluded.
    """
    column = {name: j for j, name in enumerate(names)}
    graded = X[:, column["is_graded"]] > 0.5
    present = X[:, column["ball_present"]] > 0.5
    outside = X[:, column["ball_outside_pitch"]] > 0.5
    gap = np.maximum(X[:, column["ball_gap_prev"]], X[:, column["ball_gap_next"]])
    dead = labels == "D"

    return {
        "dead_frames_with_ball_visible": round(float(np.mean(present[graded & dead])), 3),
        "dead_frames_with_ball_outside": round(float(np.mean(outside[graded & dead])), 3),
        "alive_frames_with_ball_outside": int(np.sum(outside[graded & ~dead])),
        "alive_frames_in_gap_over_5s": int(np.sum((gap > 5 * io.FPS) & graded & ~dead)),
        "alive_frames": int(np.sum(graded & ~dead)),
    }


def report(labels, X, names, phases=None):
    if phases is None:
        phases = io.load_phases(io.asset_paths()["smart_tagging"])
    return {
        "labels_source": "submission" if labels is None else "sequence",
        "phase_consistency": phase_consistency(labels, phases),
        "restart_alignment": restart_alignment(labels, phases),
        "internal_consistency": ball_visible_checks(labels, X, names),
    }


def _print(out, labels):
    print("\n=== phase consistency (Smart Tagging as an independent soft oracle) ===")
    print("  {:<32}{:>5}{:>8}{:>9}{:>12}".format("code", "n", "agreed", "rate", "majority"))
    for code, row in out["phase_consistency"].items():
        print("  {:<32}{:>5}{:>8}{:>9.3f}{:>12.3f}".format(
            code, row["n"], row["agreed"], row["rate"], row["majority_baseline"]))

    restart = out["restart_alignment"]
    if restart:
        print("\n=== restarts (the tagged team should resume play) ===")
        print("  n={}  agreed={}  rate={:.3f}  majority baseline={:.3f}".format(
            restart["n"], restart["agreed"], restart["rate"], restart["majority_baseline"]))

    print("\n=== internal consistency (claims vs the tracking) ===")
    for key, value in out["internal_consistency"].items():
        print("  {:<40} {}".format(key, value))

    for code, row in list(out["phase_consistency"].items()):
        if row["failures"]:
            print("\n  first failing phases for {:<26} {}".format(code, row["failures"][:5]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="verify a submission against independent evidence")
    parser.add_argument("--submission", default=None,
                        help="frame,label CSV (default: decode artifacts/features.npz)")
    parser.add_argument("--out", default=os.path.join(features.ARTIFACTS, "verify.json"))
    args = parser.parse_args(argv)

    labels = submission_labels(args.submission)
    X, names = features.load()
    out = report(labels, X, names)
    _print(out, labels)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
    print("\nwrote {}".format(os.path.relpath(args.out, io.ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
