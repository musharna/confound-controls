"""Ablation-control tests, including the inert-ablation refusal."""

import numpy as np
import pytest

from confound_controls import (
    ABLATION_INCONCLUSIVE,
    STRUCTURE_DEPENDENT,
    STRUCTURE_INDEPENDENT,
    ablation_control,
    assert_ablation_changed_input,
)


def _case(n=400, seed=0, survives=False):
    """Real scores separate; ablated ones either collapse or survive."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    real = y + rng.normal(0, 0.4, n)
    ablated = (y + rng.normal(0, 0.4, n)) if survives else rng.normal(0, 1, n)
    return y, real, ablated


def test_a_collapsing_ablation_is_structure_dependent():
    y, real, ablated = _case()
    res = ablation_control(y, real, ablated)
    assert res.verdict == STRUCTURE_DEPENDENT
    assert res.ablated_includes_chance
    assert res.delta > 0 and res.delta_lo > 0


def test_a_surviving_ablation_is_structure_independent():
    y, real, ablated = _case(survives=True)
    res = ablation_control(y, real, ablated)
    assert res.verdict == STRUCTURE_INDEPENDENT
    assert res.ablated_lo > 0.5


def test_the_two_cases_are_actually_different():
    # Positive control for the pair above.
    y, _, collapsed = _case()
    _, _, survived = _case(survives=True)
    from confound_controls import bootstrap_auroc

    assert bootstrap_auroc(y, collapsed).point < 0.6
    assert bootstrap_auroc(y, survived).point > 0.9


def test_an_inert_ablation_is_refused_on_the_INPUT():
    """The failure the source guarded against, generalised.

    A broken ablation leaves the input unchanged, so the scores match exactly,
    the delta is zero, and the control reports the model as surviving -- the
    strongest possible result, obtained by doing nothing.
    """
    real = np.arange(40, dtype=float).reshape(10, 4)
    with pytest.raises(ValueError, match="identical to the real input"):
        assert_ablation_changed_input(real, real.copy())

    # Positive control: a genuinely altered input passes the same check, so the
    # guard is discriminating rather than always raising.
    altered = real.copy()
    altered[0, 0] += 1.0
    assert_ablation_changed_input(real, altered)


def test_the_inert_guard_works_on_non_float_inputs():
    # Sequences, tokens, labels -- np.allclose would raise on these.
    real = np.array(["ACGT", "TTGA", "CCGG"])
    with pytest.raises(ValueError, match="identical to the real input"):
        assert_ablation_changed_input(real, real.copy())
    assert_ablation_changed_input(real, np.array(["ACGT", "TTGA", "GGCC"]))


def test_shape_mismatch_is_an_error_not_an_ablation():
    with pytest.raises(ValueError, match="shapes differ"):
        assert_ablation_changed_input(np.zeros((4, 2)), np.zeros((4, 3)))


def test_an_inert_ablation_would_otherwise_read_as_survival():
    # Demonstrates WHY the input guard exists: identical scores produce exactly
    # the reassuring answer. Nothing in the score-level statistics can catch it,
    # which is why the check belongs on the input.
    y, real, _ = _case()
    res = ablation_control(y, real, real)
    assert res.delta == 0.0
    assert res.verdict == STRUCTURE_INDEPENDENT, (
        "an inert ablation reports the model as surviving -- this is the "
        "failure mode assert_ablation_changed_input exists to prevent"
    )


def test_a_wide_interval_is_inconclusive_when_opted_in():
    rng = np.random.default_rng(7)
    y = np.r_[np.ones(10, int), np.zeros(10, int)]
    real = y + rng.normal(0, 1.0, 20)
    ablated = rng.normal(0, 1, 20)
    res = ablation_control(y, real, ablated, n=500, max_ci_width=0.05)
    assert res.verdict == ABLATION_INCONCLUSIVE


def test_degenerate_inputs_raise():
    y, real, ablated = _case()
    with pytest.raises(ValueError, match="lengths differ"):
        ablation_control(y, real[:-1], ablated)
    with pytest.raises(ValueError, match="both classes"):
        ablation_control(np.ones(10), np.arange(10.0), np.arange(10.0))
