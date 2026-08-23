#!/usr/bin/env python3
"""
Patch-level blind-test scoring, reimplemented from Aziz's shared test suite
(~/Downloads/share_nora_round2/tests/test_score_blind_round1.py) rather than
from an implementation he sent - only the test spec was shared (methodology-
sharing, not code-sharing: "this is everything we release; per-patch truth
stays sealed"). load_detections/normalize_detection/score_round1 below are
built to satisfy that spec exactly - see tests/test_score_blind_round1.py in
THIS repo, which runs Aziz's own test file against this implementation
before any real number gets trusted from it.

Used for the reverse-leg scoring (Aziz's frozen detector run on our own
round2_reverse_leg blind volume, against our private answer key) - "score it
patch-level and aggregates-only, same as I did for you."
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def normalize_detection(det):
    if 'blind_id' in det:
        pid = det['blind_id']
    elif 'patch_id' in det:
        pid = det['patch_id']
    else:
        raise ValueError("detection has no patch id (expected 'blind_id' or 'patch_id')")

    if 'is_dhi' not in det:
        raise ValueError(f"detection {pid!r} has no 'is_dhi'")
    if not isinstance(det['is_dhi'], bool):
        raise ValueError(f"detection {pid!r}: expected a JSON boolean for is_dhi, got {det['is_dhi']!r}")

    if 'confidence' not in det:
        raise ValueError(f"detection {pid!r} has no 'confidence'")
    confidence = det['confidence']
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"detection {pid!r}: confidence {confidence} outside [0, 1]")

    tier = det.get('tier', det.get('predicted_tier'))
    return {'id': pid, 'is_dhi': det['is_dhi'], 'confidence': confidence, 'tier': tier}


def load_detections(doc):
    raw = doc['detections'] if isinstance(doc, dict) and 'detections' in doc else doc
    dets = [normalize_detection(d) for d in raw]
    seen = set()
    for d in dets:
        if d['id'] in seen:
            raise ValueError(f"duplicate detection for id {d['id']!r}")
        seen.add(d['id'])
    return dets


def _wilson_interval(successes, n, z=1.959963984540054):
    """95% Wilson score interval - stable at small n/extreme rates, unlike
    the normal approximation. Returns None if n == 0 (no denominator)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - half) / denom, (centre + half) / denom)


def score_round1(key_rows, dets):
    key_by_id = {r['patch_id']: r for r in key_rows}
    det_by_id = {d['id']: d for d in dets}

    unanswered = set(key_by_id) - set(det_by_id)
    if unanswered:
        raise ValueError(f"{len(unanswered)} unanswered patch id(s): {sorted(unanswered)}")
    unknown = set(det_by_id) - set(key_by_id)
    if unknown:
        raise ValueError(f"detections reference {len(unknown)} unknown ids: {sorted(unknown)}")

    ids = list(key_by_id)
    y_true = np.array([key_by_id[i]['is_dhi'] for i in ids])
    y_conf = np.array([det_by_id[i]['confidence'] for i in ids])
    y_pred = np.array([det_by_id[i]['is_dhi'] for i in ids])

    roc_auc = float(roc_auc_score(y_true, y_conf)) if len(set(y_true)) > 1 else None
    pr_auc = float(average_precision_score(y_true, y_conf)) if len(set(y_true)) > 1 else None

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)

    recall_by_tier = {}
    for tier in sorted({key_by_id[i]['tier'] for i in ids if key_by_id[i]['is_dhi']}):
        tier_ids = [i for i in ids if key_by_id[i]['is_dhi'] and key_by_id[i]['tier'] == tier]
        hits = sum(det_by_id[i]['is_dhi'] for i in tier_ids)
        recall_by_tier[f'tier{tier}' if isinstance(tier, int) else str(tier)] = {
            'rate': hits / len(tier_ids), 'n': len(tier_ids),
            'wilson_95ci': _wilson_interval(hits, len(tier_ids)),
        }

    fp_rate_by_kind = {}
    for kind in sorted({key_by_id[i]['kind'] for i in ids if not key_by_id[i]['is_dhi']}):
        kind_ids = [i for i in ids if not key_by_id[i]['is_dhi'] and key_by_id[i]['kind'] == kind]
        flagged = sum(det_by_id[i]['is_dhi'] for i in kind_ids)
        fp_rate_by_kind[kind] = {
            'rate': flagged / len(kind_ids), 'n': len(kind_ids),
            'wilson_95ci': _wilson_interval(flagged, len(kind_ids)),
        }

    k = int(np.sum(y_true))
    top_k_ids = [ids[i] for i in np.argsort(-y_conf)[:k]] if k > 0 else []
    recall_at_k = (sum(key_by_id[i]['is_dhi'] for i in top_k_ids) / k) if k > 0 else None

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'at_her_threshold': {
            'recall': recall, 'precision': precision, 'f1': f1,
            'recall_95ci': _wilson_interval(tp, tp + fn) if (tp + fn) > 0 else None,
            'precision_95ci': _wilson_interval(tp, tp + fp) if (tp + fp) > 0 else None,
            'recall_by_tier': recall_by_tier,
            'fp_rate_by_kind': fp_rate_by_kind,
            'tp': tp, 'fp': fp, 'fn': fn,
        },
        'rank_based': {'k': k, 'recall_at_k': recall_at_k},
        '_per_patch': [
            {'id': i, 'is_dhi_true': bool(key_by_id[i]['is_dhi']), 'kind': key_by_id[i]['kind'],
             'tier': key_by_id[i]['tier'], 'confidence': float(det_by_id[i]['confidence']),
             'is_dhi_pred': bool(det_by_id[i]['is_dhi'])}
            for i in ids
        ],
    }


def main():
    if len(sys.argv) != 3:
        print('usage: score_blind_round1.py <answer_key.csv> <detections.json>', file=sys.stderr)
        sys.exit(1)
    key_path, det_path = sys.argv[1], sys.argv[2]

    key_df = pd.read_csv(key_path)
    key_rows = key_df[['blind_id', 'is_dhi', 'kind', 'tier']].rename(
        columns={'blind_id': 'patch_id'}).to_dict('records')

    doc = json.loads(Path(det_path).read_text())
    dets = load_detections(doc)

    result = score_round1(key_rows, dets)
    public = {k: v for k, v in result.items() if k != '_per_patch'}
    print(json.dumps(public, indent=2))


if __name__ == '__main__':
    main()
