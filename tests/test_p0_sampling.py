import numpy as np
import pandas as pd

from src.dhi_pipeline.dataset import _balanced_sites
from src.dhi_pipeline.scenarios import TIER_RANGES, sample_positive_scenario


def test_each_class_can_be_balanced_across_the_same_sites():
    sites = pd.DataFrame({
        'inline': [100, 200, 300],
        'crossline': [400, 500, 600],
    })
    rng = np.random.default_rng(7)
    classes_at_site = {tuple(row): set() for row in sites.to_numpy()}

    for tier in TIER_RANGES:
        for site in _balanced_sites(sites, len(sites), rng):
            _, label = sample_positive_scenario(
                tier, rng, sites, velocity_mps=2200, freq_hz=60, site=site,
            )
            classes_at_site[(label['il_center'], label['xl_center'])].add(label['kind'])

    assert all(classes == set(TIER_RANGES) for classes in classes_at_site.values())


def test_balanced_sites_round_robins_when_more_examples_are_requested():
    sites = pd.DataFrame({'inline': [1, 2, 3], 'crossline': [4, 5, 6]})
    selected = _balanced_sites(sites, 8, np.random.default_rng(1))
    counts = pd.Series([site['inline'] for site in selected]).value_counts()
    assert counts.max() - counts.min() <= 1
