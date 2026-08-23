# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""
Round 2, P1 item 3 (Aziz, ROUND2_PLAN.md): a global 0.5 threshold on the raw
XGBoost probability has failed twice for the same underlying reason - LOSO
cross-validation and the round-1 blind exchange both showed per-volume score
offsets, where the same true class sits at a different point on the raw
probability scale depending on which volume/survey produced the patch. The
absolute score isn't comparable across volumes even when relative ordering
within a volume still is.

rank_quantile below reports each patch's percentile rank *within its own
batch* instead - label-free (needs only the batch's own raw scores, nothing
about ground truth), so it's legitimate to compute at genuine blind-scoring
time, unlike anything that would need calibration against known labels.
"""

import numpy as np


def rank_quantile(scores):
    """
    Percentile rank of each score within `scores` (same batch/volume) - 0.0
    for the lowest score in the batch, 1.0 for the highest, ties averaged.
    A single-element batch returns 0.5 (undefined ordering, neutral value)
    rather than dividing by zero.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) <= 1:
        return np.full(scores.shape, 0.5)

    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    # average-tie handling: equal scores get the mean of the ranks they span
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()

    return ranks / (len(scores) - 1)
