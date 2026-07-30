"""Sequence ablation: the shuffle, the span controls, and the cluster bootstrap."""

import subprocess
import sys

import numpy as np
import pytest

from confound_controls import (
    CONFIRMED,
    NOT_CONFIRMED,
    GroupedDelta,
    confirm_knockout,
    dinucleotide_counts,
    dinucleotide_shuffle,
    grouped_delta_ci,
    knockout_span,
    sample_control_span,
)

SEQ = "ACGTACGTTGCAACGTTTGCAGGCATACG"


def test_dinucleotide_counts_are_preserved_exactly():
    # The entire point of this shuffle over a naive one: composition AND
    # dinucleotide frequency are held fixed while order is destroyed.
    res = dinucleotide_shuffle(SEQ, seed=1)
    assert res.changed
    assert dinucleotide_counts(res.sequence) == dinucleotide_counts(SEQ)
    assert sorted(res.sequence) == sorted(SEQ)


def test_the_shuffle_is_deterministic_ACROSS_PROCESSES():
    """The determinism bug the source author found the hard way.

    Using `set(graph)` for the vertex list made rng.choice/rng.shuffle consume
    the PRNG in a different order per process, because set iteration order
    depends on the randomized hash of the string keys. Same (seq, seed),
    different shuffle, run to run. A same-process test cannot catch it -- the
    hash seed is fixed for the life of an interpreter.
    """
    prog = (
        "import sys; sys.path.insert(0, '.');"
        "from confound_controls import dinucleotide_shuffle;"
        f"print(dinucleotide_shuffle({SEQ!r}, 7).sequence)"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", prog], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(4)
    }
    assert len(runs) == 1, (
        f"same (seq, seed) gave {len(runs)} different shuffles: {runs}"
    )


def test_a_sequence_that_cannot_be_shuffled_reports_unchanged():
    # The source returned the input silently here, which downstream is a
    # knockout that knocked nothing out.
    res = dinucleotide_shuffle("AAAAAAAA", seed=3)
    assert not res.changed
    assert res.sequence == "AAAAAAAA"
    with pytest.raises(ValueError, match="returned the input unchanged"):
        res.require_changed()


def test_require_changed_passes_on_a_real_shuffle():
    # Positive control for the raise above: the guard must also say yes.
    assert dinucleotide_shuffle(SEQ, seed=5).require_changed().changed


def test_short_sequences_are_reported_not_silently_returned():
    res = dinucleotide_shuffle("AC", seed=1)
    assert res.sequence == "AC" and not res.changed


def test_knockout_leaves_the_flanks_byte_identical():
    res = knockout_span(SEQ, 8, 20, seed=0)
    assert res.changed
    assert res.sequence[:8] == SEQ[:8]
    assert res.sequence[20:] == SEQ[20:]
    assert res.sequence[8:20] != SEQ[8:20]
    assert len(res.sequence) == len(SEQ)


def test_knockout_rejects_a_span_outside_the_sequence():
    with pytest.raises(ValueError, match="not inside a sequence"):
        knockout_span(SEQ, 5, len(SEQ) + 1, seed=1)
    with pytest.raises(ValueError, match="not inside a sequence"):
        knockout_span(SEQ, 7, 7, seed=1)


def test_a_control_span_avoids_the_real_spans_when_there_is_room():
    rng = np.random.RandomState(0)
    span = sample_control_span(length=1000, total=10, spans=[(100, 150)], rng=rng)
    assert span.disjoint
    assert span.end <= 100 or span.start >= 150
    span.require_disjoint()


def test_a_control_span_that_cannot_avoid_them_says_so():
    # The source returned an overlapping start after 50 tries with no signal,
    # so the "random control" became a partial real knockout compared against
    # itself.
    rng = np.random.RandomState(0)
    span = sample_control_span(length=30, total=25, spans=[(0, 30)], rng=rng)
    assert not span.disjoint
    with pytest.raises(ValueError, match="OVERLAPS them"):
        span.require_disjoint()


def test_control_span_rejects_a_nonpositive_length():
    with pytest.raises(ValueError, match="must be positive"):
        sample_control_span(100, 0, [], np.random.RandomState(0))


def _clustered(n_groups=12, per=8, effect=0.3, seed=0):
    """Observations correlated WITHIN group -- the situation clustering exists for.

    The effect itself carries a group-level component. Without that the delta
    is identical for every row, both bootstraps return a zero-width interval,
    and the comparison below is vacuous -- which is exactly what the first
    version of this fixture did.
    """
    rng = np.random.default_rng(seed)
    groups, real, ko = [], [], []
    for g in range(n_groups):
        shift = rng.normal(0, 1.0)                  # whole-family offset
        group_effect = effect + rng.normal(0, 0.25)  # effect varies BY family
        for _ in range(per):
            base = rng.normal(0, 0.2) + shift
            groups.append(g)
            real.append(base + group_effect + rng.normal(0, 0.05))
            ko.append(base)
    return np.array(real), np.array(ko), np.array(groups)


def test_the_cluster_bootstrap_is_wider_than_a_row_bootstrap():
    """Why grouped: rows within a paralog family are not independent.

    A row-level bootstrap treats 96 correlated observations as 96 independent
    ones and reports an interval far narrower than the data supports.
    """
    real, ko, groups = _clustered()
    grouped = grouped_delta_ci(real, ko, groups, n=1000)
    rowwise = grouped_delta_ci(real, ko, np.arange(len(real)), n=1000, min_groups=1)
    assert (grouped.hi - grouped.lo) > (rowwise.hi - rowwise.lo), (
        f"grouped {grouped.hi - grouped.lo:.4f} vs rowwise "
        f"{rowwise.hi - rowwise.lo:.4f}: clustering must not narrow the interval"
    )
    assert grouped.n_groups == 12


def test_a_real_effect_is_still_detected():
    # Positive control: widening must not mean detecting nothing.
    real, ko, groups = _clustered(effect=0.5)
    assert grouped_delta_ci(real, ko, groups, n=1000).excludes_zero


def test_too_few_groups_raises_rather_than_reporting_an_interval():
    real, ko, groups = _clustered(n_groups=3)
    with pytest.raises(ValueError, match="distinct group"):
        grouped_delta_ci(real, ko, groups, n=200)
    # ...and the override still works, so the guard is a choice not a wall.
    assert grouped_delta_ci(real, ko, groups, n=200, min_groups=1).n_groups == 3


def test_mismatched_shapes_raise():
    real, ko, groups = _clustered()
    with pytest.raises(ValueError, match="shapes differ"):
        grouped_delta_ci(real[:-1], ko, groups)


def _gd(mean, lo, hi):
    return GroupedDelta(mean, lo, hi, 12)


def test_confirmation_needs_all_three_clauses():
    pooled = _gd(0.30, 0.10, 0.50)
    weak_control = _gd(0.05, -0.05, 0.15)
    assert confirm_knockout(pooled, weak_control) == CONFIRMED

    # interval touches zero -> not confirmed
    assert confirm_knockout(_gd(0.30, -0.02, 0.60), weak_control) == NOT_CONFIRMED
    # drop is real but no larger than a length-matched control knockout: the
    # clause that stops "any perturbation hurts" reading as "this element matters"
    assert confirm_knockout(pooled, _gd(0.40, 0.20, 0.60)) == NOT_CONFIRMED
    # wrong direction
    assert confirm_knockout(_gd(-0.30, -0.50, -0.10), weak_control) == NOT_CONFIRMED


def test_the_silent_fallback_case_is_not_rare_on_real_spans():
    """Measured, not assumed.

    The source called the identity fallback "rare" in a comment and returned it
    silently. On this real 29-mer, knocking out [8:20] leaves the span UNCHANGED
    for 2 of the first 8 seeds -- so a study looping over instances with
    sequential seeds gets inert knockouts at a rate nothing was reporting.
    """
    outcomes = {s: knockout_span(SEQ, 8, 20, seed=s).changed for s in range(8)}
    unchanged = [s for s, ch in outcomes.items() if not ch]
    assert unchanged == [2, 3], outcomes
    for s in unchanged:
        with pytest.raises(ValueError, match="returned the input unchanged"):
            knockout_span(SEQ, 8, 20, seed=s).require_changed()
