"""Paired-delta tests against data with a known answer."""

import numpy as np
import pytest

from confound_controls import (
    ADDS,
    HARMS,
    NO_EFFECT,
    UNDERPOWERED,
    incremental_validity,
    paired_delta_auroc,
)


def _scores(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    weak = y + rng.normal(0, 1.6, n)
    strong = y + rng.normal(0, 0.4, n)
    return y, weak, strong


def test_a_better_model_reports_adds_signal():
    y, weak, strong = _scores()
    res = paired_delta_auroc(y, weak, strong)
    assert res.verdict == ADDS
    assert res.delta > 0 and res.lo > 0
    assert res.excludes_zero


def test_a_worse_model_reports_harms_not_no_effect():
    # The source collapsed this into "no-added-signal". An augmented model that
    # is reliably WORSE is a finding, not an absence of one.
    y, weak, strong = _scores()
    res = paired_delta_auroc(y, strong, weak)  # augmented is the weak one
    assert res.verdict == HARMS
    assert res.hi < 0


def test_identical_scores_report_no_effect():
    y, weak, _ = _scores()
    res = paired_delta_auroc(y, weak, weak)
    assert res.verdict == NO_EFFECT
    assert res.delta == 0.0
    assert not res.excludes_zero


def test_a_wide_interval_is_underpowered_only_when_opted_in():
    # 20 rows: the delta interval is far too wide to distinguish anything.
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(10, int), np.zeros(10, int)]
    a = y + rng.normal(0, 1.0, 20)
    b = y + rng.normal(0, 1.0, 20)
    default = paired_delta_auroc(y, a, b, n=500)
    assert default.verdict in (NO_EFFECT, ADDS, HARMS)
    wide = paired_delta_auroc(y, a, b, n=500, max_ci_width=0.05)
    assert wide.verdict == UNDERPOWERED


def test_the_delta_is_paired_not_two_independent_intervals():
    """The reason this function exists.

    Two independent intervals on two AUROCs overlap far more often than the
    paired interval on their difference includes zero, because the pair shares
    resampled rows and the common variance cancels. If this ever inverts, the
    implementation has stopped pairing.
    """
    from confound_controls import bootstrap_auroc

    # Tuned so the two independent intervals genuinely overlap -- measured
    # A=[0.632,0.778] B=[0.760,0.874] -- while the paired delta still excludes
    # zero at +0.021. A wider gap would separate the independent intervals too
    # and the test would prove nothing.
    rng = np.random.default_rng(3)
    n = 200
    y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    weak = y + rng.normal(0, 1.0, n)
    strong = y + rng.normal(0, 0.85, n)
    ci_a = bootstrap_auroc(y, weak, n=800)
    ci_b = bootstrap_auroc(y, strong, n=800)
    independent_overlap = ci_a.hi >= ci_b.lo and ci_b.hi >= ci_a.lo

    paired = paired_delta_auroc(y, weak, strong, n=800)
    assert independent_overlap, "fixture no longer exercises the distinction"
    assert paired.lo > 0, (
        "the paired delta must resolve a difference the two independent intervals cannot"
    )


def test_degenerate_inputs_raise():
    y, weak, strong = _scores()
    with pytest.raises(ValueError, match="lengths differ"):
        paired_delta_auroc(y, weak[:-1], strong)
    with pytest.raises(ValueError, match="both classes"):
        paired_delta_auroc(np.ones(10), np.arange(10.0), np.arange(10.0))


def test_incremental_validity_detects_a_feature_that_adds_nothing():
    rng = np.random.default_rng(1)
    n = 500
    y_tr = rng.integers(0, 2, n)
    y_te = rng.integers(0, 2, n)
    conf_tr = np.column_stack([y_tr + rng.normal(0, 0.6, n), rng.normal(0, 1, n)])
    conf_te = np.column_stack([y_te + rng.normal(0, 0.6, n), rng.normal(0, 1, n)])
    noise_tr, noise_te = rng.normal(0, 1, n), rng.normal(0, 1, n)

    res = incremental_validity(conf_tr, conf_te, noise_tr, noise_te, y_tr, y_te, n=500)
    assert res.verdict in (NO_EFFECT, HARMS), res

    # Positive control in the same test: a feature that DOES carry the label is
    # detected by the same call, so the null above is not a dead harness.
    good_tr, good_te = y_tr + rng.normal(0, 0.3, n), y_te + rng.normal(0, 0.3, n)
    res2 = incremental_validity(conf_tr, conf_te, good_tr, good_te, y_tr, y_te, n=500)
    assert res2.verdict == ADDS, res2


def test_incremental_validity_rejects_mismatched_row_counts():
    rng = np.random.default_rng(2)
    conf = rng.normal(0, 1, (50, 2))
    y = rng.integers(0, 2, 50)
    with pytest.raises(ValueError, match="train rows differ"):
        incremental_validity(conf, conf, rng.normal(0, 1, 49), rng.normal(0, 1, 50), y, y)
