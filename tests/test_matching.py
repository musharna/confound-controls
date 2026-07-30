import numpy as np
import pytest

from confound_controls import match_negatives


def test_one_to_one_match_when_the_pool_is_large_enough():
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 1, (10, 2))
    neg = rng.normal(0, 1, (100, 2))
    res = match_negatives(pos, [f"n{i}" for i in range(100)], neg)
    assert res.complete
    assert res.n_matched == 10
    assert len(set(res.matched_ids)) == 10, "matching must be without replacement"


def test_an_exhausted_pool_is_reported_not_hidden():
    # The extracted original returned a short list and said nothing. A caller
    # comparing AUROC before and after matching could not tell a clean 1:1
    # design from one that quietly dropped a third of its positives.
    pos = np.arange(10, dtype=float).reshape(10, 1)
    neg = np.arange(3, dtype=float).reshape(3, 1)
    res = match_negatives(pos, ["a", "b", "c"], neg)
    assert not res.complete
    assert res.n_matched == 3
    assert len(res.unmatched_positions) == 7
    with pytest.raises(ValueError, match="matched only 3 of 10"):
        res.require_complete()


def test_require_complete_returns_self_when_complete():
    # Positive control for the raise above: the guard must also say YES, or a
    # harness that raised unconditionally would look identical.
    pos = np.zeros((2, 1))
    neg = np.zeros((5, 1))
    res = match_negatives(pos, list("abcde"), neg).require_complete()
    assert res.complete and res.n_matched == 2


def test_an_empty_negative_pool_matches_nothing_and_says_so():
    res = match_negatives(np.zeros((3, 1)), [], np.zeros((0, 1)))
    assert not res.complete
    assert res.n_matched == 0
    assert res.unmatched_positions == [0, 1, 2]


def test_a_constant_confound_does_not_produce_nan_distances():
    # sd == 0 on a column would divide by zero, and nan distances match at
    # random -- which is indistinguishable from a successful match.
    pos = np.column_stack([np.arange(5.0), np.ones(5)])
    neg = np.column_stack([np.arange(20.0), np.ones(20)])
    res = match_negatives(pos, [f"n{i}" for i in range(20)], neg)
    assert res.complete
    assert not any(m is None for m in res.matched_ids)


def test_matching_picks_the_nearest_available_negative():
    # Behavioural, not just structural: a positive at 100 must take the
    # negative at 100, not an arbitrary one.
    pos = np.array([[100.0]])
    neg = np.array([[0.0], [100.0], [200.0]])
    res = match_negatives(pos, ["far_lo", "near", "far_hi"], neg)
    assert res.matched_ids == ["near"]


def test_shape_mismatches_raise():
    with pytest.raises(ValueError, match="confound count differs"):
        match_negatives(np.zeros((2, 3)), ["a", "b"], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="neg_ids has 1 entries"):
        match_negatives(np.zeros((2, 2)), ["a"], np.zeros((2, 2)))
