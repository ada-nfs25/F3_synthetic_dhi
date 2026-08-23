"""Tests for scripts/score_blind_round1.py (round-one blind scoring).

The scoring script is the LAST code between her detections and the numbers
both reports will quote -- a silent join or metric bug here corrupts the
project's headline result, so its pure functions get the same test
discipline as dhi_lib.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_blind_round1 import (  # noqa: E402
    load_detections, normalize_detection, score_round1,
)


def _key_rows():
    """4 positives (tiers 1/2/3/4), 2 hard negatives, 2 backgrounds."""
    rows = []
    for i, tier in enumerate((1, 2, 3, 4)):
        rows.append({"patch_id": f"p{i}", "kind": "bright_spot", "tier": tier,
                     "is_dhi": True})
    rows.append({"patch_id": "n0", "kind": "tuning", "tier": None, "is_dhi": False})
    rows.append({"patch_id": "n1", "kind": "single_reflector", "tier": None,
                 "is_dhi": False})
    rows.append({"patch_id": "b0", "kind": "background", "tier": None,
                 "is_dhi": False})
    rows.append({"patch_id": "b1", "kind": "background", "tier": None,
                 "is_dhi": False})
    return rows


def _det(pid, is_dhi, conf, **extra):
    return {"patch_id": pid, "is_dhi": is_dhi, "confidence": conf, **extra}


def test_normalize_accepts_v10_dummy_field_names():
    d = normalize_detection({"blind_id": "p0", "is_dhi": True,
                             "confidence": 0.82, "predicted_tier": None})
    assert d == {"id": "p0", "is_dhi": True, "confidence": 0.82, "tier": None}


def test_normalize_rejects_non_boolean_is_dhi():
    with pytest.raises(ValueError, match="expected a JSON boolean"):
        normalize_detection({"patch_id": "p0", "is_dhi": 1, "confidence": 0.5})


def test_normalize_rejects_missing_fields_loudly():
    with pytest.raises(ValueError, match="no patch id"):
        normalize_detection({"is_dhi": True, "confidence": 0.5})
    with pytest.raises(ValueError, match="no 'confidence'"):
        normalize_detection({"patch_id": "p0", "is_dhi": True})
    with pytest.raises(ValueError, match="outside"):
        normalize_detection({"patch_id": "p0", "is_dhi": True, "confidence": 1.5})


def test_load_detections_unwraps_dict_and_rejects_duplicates():
    doc = {"detections": [_det("p0", True, 0.9), _det("p1", False, 0.1)]}
    assert len(load_detections(doc)) == 2
    with pytest.raises(ValueError, match="duplicate"):
        load_detections([_det("p0", True, 0.9), _det("p0", True, 0.9)])


def test_score_requires_exact_coverage():
    dets = [normalize_detection(_det(r["patch_id"], False, 0.1))
            for r in _key_rows()[:-1]]  # one unanswered
    with pytest.raises(ValueError, match="unanswered"):
        score_round1(_key_rows(), dets)
    dets = [normalize_detection(_det(r["patch_id"], False, 0.1))
            for r in _key_rows()] + [normalize_detection(_det("ghost", True, 0.9))]
    with pytest.raises(ValueError, match="unknown ids"):
        score_round1(_key_rows(), dets)


def test_score_perfect_detector():
    dets = [normalize_detection(_det(r["patch_id"], r["is_dhi"],
                                     0.9 if r["is_dhi"] else 0.1))
            for r in _key_rows()]
    s = score_round1(_key_rows(), dets)
    assert s["roc_auc"] == 1.0 and s["pr_auc"] == 1.0
    at = s["at_her_threshold"]
    assert at["recall"] == 1.0 and at["precision"] == 1.0 and at["f1"] == 1.0
    assert all(v["rate"] == 1.0 for v in at["recall_by_tier"].values())
    assert all(v["rate"] == 0.0 for v in at["fp_rate_by_kind"].values())
    assert s["rank_based"] == {"k": 4, "recall_at_k": 1.0}


def test_score_known_mixed_case():
    """Hand-computed: misses tier 1, false-positives on tuning; confidences
    rank the tuning FP above the missed tier-1 positive."""
    conf = {"p0": 0.30, "p1": 0.80, "p2": 0.85, "p3": 0.90,   # p0 missed
            "n0": 0.60, "n1": 0.20, "b0": 0.10, "b1": 0.05}   # n0 FP
    dets = [normalize_detection(_det(pid, c >= 0.5, c)) for pid, c in conf.items()]
    s = score_round1(_key_rows(), dets)
    at = s["at_her_threshold"]
    assert at["recall"] == 0.75                      # 3/4
    assert at["precision"] == 0.75                   # 3 TP / (3 TP + 1 FP)
    assert at["recall_by_tier"]["tier1"]["rate"] == 0.0
    assert at["recall_by_tier"]["tier4"]["rate"] == 1.0
    assert at["fp_rate_by_kind"]["tuning"]["rate"] == 1.0
    assert at["fp_rate_by_kind"]["single_reflector"]["rate"] == 0.0
    assert at["fp_rate_by_kind"]["background"]["rate"] == 0.0
    # top-4 by confidence = p3, p2, p1, n0 -> 3 of 4 positives captured
    assert s["rank_based"] == {"k": 4, "recall_at_k": 0.75}
    # per-patch join is present for the private file but is a private field
    assert len(s["_per_patch"]) == 8
