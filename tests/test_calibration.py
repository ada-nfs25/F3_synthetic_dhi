# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""Round 2, P1 item 3: rank_quantile must give a comparable, label-free
ordering within a batch regardless of where the raw scores happen to sit."""
import numpy as np
import pytest

from src.dhi_pipeline.calibration import rank_quantile


def test_extremes_and_monotonicity():
    q = rank_quantile([0.1, 0.4, 0.9])
    assert q[0] == pytest.approx(0.0)
    assert q[2] == pytest.approx(1.0)
    assert q[0] < q[1] < q[2]


def test_ties_get_averaged_rank():
    q = rank_quantile([0.2, 0.5, 0.5, 0.8])
    assert q[1] == pytest.approx(q[2])
    assert q[0] < q[1] < q[3]


def test_offset_batches_land_on_the_same_scale():
    # same relative ordering, wildly different raw offsets - the whole point
    low_offset_batch = rank_quantile([0.05, 0.10, 0.15])
    high_offset_batch = rank_quantile([0.75, 0.80, 0.85])
    assert list(low_offset_batch) == pytest.approx(list(high_offset_batch))


def test_single_element_batch_is_neutral_not_a_crash():
    q = rank_quantile([0.3])
    assert q[0] == pytest.approx(0.5)
