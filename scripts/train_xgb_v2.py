#!/usr/bin/env python3
# Implementation developed with AI (Claude Code) assistance - see AI_USAGE.md.
"""
P2: LOSO-evaluate and fit XGBoost v2 on the P0-diversified dataset
(H1 + H3 + dim_spot), comparing v1's original feature definitions against
the P1-fixed ones (swept-lag doublet, survey-adaptive band_ratio) on the
SAME new data - Aziz's ROUND2_PLAN.md P1 item 1: "your earlier rejection...
was measured on the thin-bed-only training set... re-evaluate on the
diversified set from P0, not the old one."

Model architecture/hyperparameters and the LeaveOneGroupOut(site_id) scheme
are copied verbatim from irp-nfs25/notebooks/dhi_xgb_detector.ipynb's
run_loso, so only the features and training data differ from v1 - isolates
what the P0/P1 changes actually did rather than also changing the model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.score_blind_round1 import _wilson_interval  # noqa: E402

FEATURES_PATH = REPO / 'data' / 'v2_features' / 'v2_combined_features.parquet'
MODEL_OUT_PATH = REPO / 'data' / 'v2_features' / 'xgboost_dhi_14feature_v2.json'
SECONDARY_MODEL_OUT_PATH = REPO / 'data' / 'v2_features' / 'xgboost_dhi_14feature_v1style_secondary.json'
LOSO_OUT_PATH = REPO / 'data' / 'v2_features' / 'v2_loso_predictions.csv'

V2_FEATURE_COLS = [
    'envelope_mean_near_top_ratio', 'envelope_p90_near_top_ratio', 'envelope_maxabs_near_top_ratio',
    'sweetness_mean_near_top_ratio', 'sweetness_p90_near_top_ratio', 'sweetness_maxabs_near_top_ratio',
    'inst_freq_mean_near_top_ratio', 'inst_freq_p90_near_top_ratio', 'inst_freq_maxabs_near_top_ratio',
    'band_ratio_mean_near_top_ratio', 'band_ratio_p90_near_top_ratio', 'band_ratio_maxabs_near_top_ratio',
    'doublet_autocorrelation', 'signed_polarity',
]
V1_STYLE_FEATURE_COLS = [f'v1style_{c}' for c in V2_FEATURE_COLS]


def run_loso(X, y, groups):
    """Verbatim hyperparameters/scheme from irp-nfs25/notebooks/dhi_xgb_detector.ipynb's run_loso."""
    Xs, yv = X.to_numpy(), y.to_numpy()
    logo = LeaveOneGroupOut()
    probs = np.zeros(len(yv), dtype=float)

    for train_idx, test_idx in logo.split(Xs, yv, groups=groups):
        y_train = yv[train_idx]
        n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
        fold_model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
            scale_pos_weight=n_neg / n_pos, eval_metric='logloss', random_state=1, n_jobs=-1,
        )
        fold_model.fit(Xs[train_idx], y_train)
        probs[test_idx] = fold_model.predict_proba(Xs[test_idx])[:, 1]

    preds = (probs >= 0.5).astype(int)
    return {
        'roc_auc': roc_auc_score(yv, probs), 'pr_auc': average_precision_score(yv, probs),
        'recall': recall_score(yv, preds), 'precision': precision_score(yv, preds),
        'f1': f1_score(yv, preds),
    }, probs, preds


def breakdown_by_tier_and_kind(df, preds):
    recall_by_tier = {}
    for tier in sorted(df.loc[df.is_dhi, 'tier'].unique()):
        sel = df.is_dhi & (df.tier == tier)
        hits = int(preds[sel.to_numpy()].sum())
        n = int(sel.sum())
        recall_by_tier[tier] = {'n': n, 'rate': hits / n, 'wilson_95ci': _wilson_interval(hits, n)}

    fp_rate_by_kind = {}
    for kind in sorted(df.loc[~df.is_dhi, 'kind'].unique()):
        sel = (~df.is_dhi) & (df.kind == kind)
        flagged = int(preds[sel.to_numpy()].sum())
        n = int(sel.sum())
        fp_rate_by_kind[kind] = {'n': n, 'rate': flagged / n, 'wilson_95ci': _wilson_interval(flagged, n)}

    return recall_by_tier, fp_rate_by_kind


def main():
    df = pd.read_parquet(FEATURES_PATH)
    y = df['is_dhi'].astype(bool)
    groups = df['site_id']
    print(f'{len(df)} examples, {groups.nunique()} LOSO sites, '
          f'{y.sum()} positive / {(~y).sum()} negative')

    print('\n=== v1-style features (fixed band_ratio, lag-1 doublet) on P0-diversified data ===')
    v1_metrics, v1_probs, v1_preds = run_loso(df[V1_STYLE_FEATURE_COLS], y, groups)
    print(json.dumps(v1_metrics, indent=2))

    print('\n=== v2 features (adaptive band_ratio, swept-lag doublet) on P0-diversified data ===')
    v2_metrics, v2_probs, v2_preds = run_loso(df[V2_FEATURE_COLS], y, groups)
    print(json.dumps(v2_metrics, indent=2))

    v2_recall_by_tier, v2_fp_rate_by_kind = breakdown_by_tier_and_kind(df, v2_preds)
    v1_recall_by_tier, v1_fp_rate_by_kind = breakdown_by_tier_and_kind(df, v1_preds)

    print('\nrecall by tier, v1-style vs v2 (n, rate):')
    for tier in v2_recall_by_tier:
        v1r, v2r = v1_recall_by_tier[tier], v2_recall_by_tier[tier]
        print(f'  {tier:20s} n={v2r["n"]:4d}  v1-style={v1r["rate"]:.3f}  v2={v2r["rate"]:.3f}')

    print('\nfp rate by kind, v1-style vs v2 (n, rate):')
    for kind in v2_fp_rate_by_kind:
        v1r, v2r = v1_fp_rate_by_kind[kind], v2_fp_rate_by_kind[kind]
        print(f'  {kind:32s} n={v2r["n"]:4d}  v1-style={v1r["rate"]:.3f}  v2={v2r["rate"]:.3f}')

    print('\nv2 recall by tier (full, with CIs):')
    print(json.dumps(v2_recall_by_tier, indent=2))
    print('\nv2 fp rate by kind (full, with CIs):')
    print(json.dumps(v2_fp_rate_by_kind, indent=2))

    n_pos, n_neg = y.sum(), (~y).sum()
    final_model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
        scale_pos_weight=n_neg / n_pos, eval_metric='logloss', random_state=1, n_jobs=-1,
    )
    final_model.fit(df[V2_FEATURE_COLS], y.astype(int))
    final_model.save_model(str(MODEL_OUT_PATH))
    print(f'\nfit final PRIMARY model (v2/P1 features) on all {len(df)} examples, saved to {MODEL_OUT_PATH}')

    # Declared secondary (Aziz, round-2 decision): same P0-diversified data,
    # v1-style features (fixed band_ratio, lag-1 doublet) - same hyperparameters
    # and full-data-fit discipline as the primary, so the two are comparable as
    # a clean ablation rather than differing in anything but the feature set.
    secondary_model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
        scale_pos_weight=n_neg / n_pos, eval_metric='logloss', random_state=1, n_jobs=-1,
    )
    # rename to the plain (unprefixed) names before fitting: the 'v1style_'
    # prefix only exists to keep the two variants distinct as separate
    # columns during the LOSO ablation above - the saved model itself must
    # expose the same feature names as the primary (both operate on "the
    # same 14 feature slots", just computed differently), or a scorer that
    # loads both models and feeds them the same feature dict (e.g.
    # score_round2_blind_set.py) breaks on a feature-name mismatch at
    # blind-scoring time, not at this training step where it's cheap to catch.
    X_secondary = df[V1_STYLE_FEATURE_COLS].rename(
        columns=dict(zip(V1_STYLE_FEATURE_COLS, V2_FEATURE_COLS)))
    secondary_model.fit(X_secondary, y.astype(int))
    secondary_model.save_model(str(SECONDARY_MODEL_OUT_PATH))
    print(f'fit final SECONDARY model (v1-style features) on all {len(df)} examples, '
          f'saved to {SECONDARY_MODEL_OUT_PATH}')

    loso_output = df[['global_id', 'dataset_source', 'is_dhi', 'kind', 'tier',
                       'il_center', 'xl_center', 'site_id']].copy()
    loso_output['loso_probability'] = v2_probs
    loso_output['loso_prediction'] = v2_preds.astype(bool)
    loso_output['v1style_loso_probability'] = v1_probs
    loso_output['v1style_loso_prediction'] = v1_preds.astype(bool)
    loso_output.to_csv(LOSO_OUT_PATH, index=False)
    print(f'wrote per-patch LOSO predictions to {LOSO_OUT_PATH}')


if __name__ == '__main__':
    main()
