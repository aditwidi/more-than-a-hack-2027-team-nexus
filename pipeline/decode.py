"""Decoding: turn emissions into a possession sequence.

This is D5 of `docs/references.md` and the part that actually moves the ranking metric. Frame-level
accuracy and possession-level F1 disagree on purpose: a system can be right on ~90% of frames and
still report three possessions where there was one. The metric matches possessions one-to-one, so
**false possessions are the expensive error** and a boundary a second late is nearly free.

Two stages, both deliberate:

1. **Viterbi** over {H, A, D} with asymmetric switch costs: changing team (H<->A) is expensive,
   entering or leaving dead ball is cheaper. This places boundaries where the evidence changes
   rather than at the centre of a smoothing window, which is what a centred majority vote cannot do
   -- and it is why `baseline_smooth50` scores 26 possessions against 17 real ones.
2. **Structural cleanup** with minimum dwell times. Short dead stretches are absorbed back into the
   possession around them (CHALLENGE.md: possession stays with the last team while the ball is
   loose or contested), and short team runs are absorbed into a neighbour. Nothing here merges the
   same team *across* a dead stretch: `H H D D H H` is two possessions, not one, and the metric is
   built on that.
"""

from __future__ import annotations

import numpy as np

from . import emissions, io
from games import half_video_blocks

LABELS = np.array(["H", "A", "D"])
NO_ANSWER = "X"


# ---------------------------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------------------------
def transition_matrix(params):
    """Log transition scores, rows = from, columns = to, for [H, A, D]."""
    dead = params["switch_penalty_dead"]
    team = params["switch_penalty"]
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[emissions.HOME, emissions.AWAY] = -team
    matrix[emissions.AWAY, emissions.HOME] = -team
    for state in (emissions.HOME, emissions.AWAY):
        matrix[state, emissions.DEAD] = -dead
        matrix[emissions.DEAD, state] = -dead
    return matrix


def viterbi(logits, params):
    """Best state path under constant transition costs. Returns int8 states.

    Cost is O(n_frames * 9); 199,319 frames take about a second, so an ablation sweep is cheap.
    """
    n_frames = logits.shape[0]
    trans = transition_matrix(params)

    score = np.array(logits[0], dtype=np.float64)
    back = np.zeros((n_frames, 3), dtype=np.int8)
    current = np.empty(3, dtype=np.float64)
    candidate = np.empty((3, 3), dtype=np.float64)

    for t in range(1, n_frames):
        candidate[:] = score[:, None] + trans
        best_from = np.argmax(candidate, axis=0)
        current[:] = candidate[best_from, np.arange(3)] + logits[t]
        back[t] = best_from
        score, current = current, score

    states = np.empty(n_frames, dtype=np.int8)
    states[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


# ---------------------------------------------------------------------------------------------
# Structural cleanup
# ---------------------------------------------------------------------------------------------
def runs(states):
    """(start, end, state) for every maximal run, end inclusive."""
    if len(states) == 0:
        return []
    change = np.flatnonzero(states[1:] != states[:-1]) + 1
    edges = np.concatenate(([0], change, [len(states)]))
    return [(int(edges[i]), int(edges[i + 1] - 1), int(states[edges[i]]))
            for i in range(len(edges) - 1)]


def minimum_dwell(params):
    return {emissions.HOME: int(params["min_possession_frames"]),
            emissions.AWAY: int(params["min_possession_frames"]),
            emissions.DEAD: int(params["min_dead_frames"])}


def cleanup(states, params):
    """Absorb runs shorter than the minimum dwell into their neighbours.

    Order of choices for a short run, in priority order:

    1. a neighbour with the **same** label -- pure flicker, merge it;
    2. both neighbours are dead and this run is a team -- a blip of possession inside a stoppage;
    3. both neighbours are the same team and this run is dead -- the ball was briefly lost mid-
       possession, which CHALLENGE.md says stays with the same team;
    4. otherwise the longer neighbour, which is the dominant state around this frame.

    Iterated to a fixed point, because absorbing one run can create another below the minimum.
    """
    min_dwell = minimum_dwell(params)
    states = states.copy()

    for _ in range(20):
        changed = False
        spans = runs(states)
        for i, (start, end, state) in enumerate(spans):
            length = end - start + 1
            if length >= min_dwell[state]:
                continue

            previous = spans[i - 1] if i > 0 else None
            following = spans[i + 1] if i + 1 < len(spans) else None

            target = None
            if previous and previous[2] == state:
                target = state
            elif following and following[2] == state:
                target = state
            elif (previous and following and previous[2] == following[2]
                  and previous[2] != emissions.DEAD and state == emissions.DEAD):
                target = previous[2]            # short dead stretch inside a possession
            elif (previous and following and previous[2] == following[2]
                  and previous[2] == emissions.DEAD and state != emissions.DEAD):
                target = emissions.DEAD          # blip of possession inside a stoppage
            elif previous and following:
                target = previous[2] if (previous[1] - previous[0]) >= (following[1] - following[0]) \
                    else following[2]
            elif previous:
                target = previous[2]
            elif following:
                target = following[2]
            else:
                break                            # the whole match is one short run

            states[start:end + 1] = target
            changed = True
            break
        if not changed:
            break
    return states


# ---------------------------------------------------------------------------------------------
# Full decode
# ---------------------------------------------------------------------------------------------
def decode(X, names, params=None, use_cleanup=True):
    """Emission -> Viterbi -> cleanup -> label letters, with `X` outside the graded halves."""
    p = dict(emissions.DEFAULTS)
    if params:
        p.update(params)

    logits = emissions.emission_logits(X, names, params)
    states = viterbi(logits, p)
    if use_cleanup:
        states = cleanup(states, p)

    labels = LABELS[states].astype("<U1")

    graded = X[:, emissions.column_index(names)["is_graded"]] > 0.5
    labels[~graded] = NO_ANSWER
    return labels


def write_submission(path, labels):
    """The submission format, in five lines (SUBMISSION.md)."""
    import os

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path
