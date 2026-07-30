"""Behavioural tests against synthetic data where the truth is known.

A confound battery is only worth anything if it reaches the RIGHT verdict, so
these build two datasets: one where the classifier's signal is genuinely the
label, and one where the signal is entirely a confound. The battery must
separate them.
"""

import numpy as np
import pandas as pd
import pytest

from confound_controls import (
    DRIVEN,
    INCONCLUSIVE,
    ROBUST,
    battery_passes,
    bootstrap_auroc,
    evaluate_confound,
    format_battery,
    recovery,
    run_battery,
    verdict,
)


def _frame(n_pos=150, n_neg=600, seed=0, confound_drives=False, twin_frac=0.35):
    """Synthetic data with a KNOWN answer, so the battery can be checked.

    The negative pool is built in two parts, and both are load-bearing:

      * a bulk at LOW gc, which makes gc separate the classes -- so a purely
        gc-driven score has a high UNMATCHED auroc, the thing a control has to
        knock down;
      * a `twin_frac` subset drawn from the positives' own gc distribution, so
        a near-exact gc twin exists for each positive and matching can actually
        neutralise gc.

    Get either wrong and the fixture stops testing what it claims. An all-low
    pool leaves residual gc after matching (measured: 0.99 -> 0.76, reads as
    "partial"); a pool sharing the positives' gc mean makes gc non-separating,
    so the unmatched auroc is already 0.5 and there is nothing to collapse.

    confound_drives=False -> probability tracks the LABEL (real signal)
    confound_drives=True  -> probability tracks GC only (confounded)
    """
    rng = np.random.default_rng(seed)
    n_twin = int(round(n_neg * twin_frac))
    gc = np.concatenate([
        rng.normal(0.55, 0.05, n_pos),               # positives
        rng.normal(0.55, 0.05, n_twin),              # gc-matched negatives
        rng.normal(0.40, 0.04, n_neg - n_twin),      # bulk low-gc negatives
    ])
    label = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    n = n_pos + n_neg
    other = rng.normal(0, 1, n)
    ids = [f"g{i:04d}" for i in range(n)]
    df = pd.DataFrame({"id": ids, "label": label, "gc": gc, "other": other})
    if confound_drives:
        score = gc + rng.normal(0, 0.01, n)
    else:
        score = label + rng.normal(0, 0.35, n)
    return df, dict(zip(ids, score))


def test_real_signal_survives_matching_on_an_irrelevant_confound():
    df, probs = _frame()
    anchor = bootstrap_auroc(df["label"], [probs[i] for i in df["id"]]).point
    res = evaluate_confound(df, probs, ["other"], anchor, name="other")
    assert res.verdict == ROBUST, res
    assert res.match_complete


def test_a_confound_driven_signal_dies_when_matched_on_that_confound():
    # The whole point of the battery. Probability is GC and nothing else, so
    # matching negatives to positives on GC must collapse it to chance.
    df, probs = _frame(confound_drives=True)
    anchor = bootstrap_auroc(df["label"], [probs[i] for i in df["id"]]).point
    res = evaluate_confound(df, probs, ["gc"], anchor, name="gc")
    assert res.verdict == DRIVEN, res
    assert res.ci.lo <= 0.5 <= res.ci.hi


def test_the_two_datasets_are_actually_different():
    # Positive control for the pair above: if _frame ignored confound_drives,
    # both tests could pass for the wrong reason.
    df_a, pa = _frame()
    df_b, pb = _frame(confound_drives=True)
    auc_a = bootstrap_auroc(df_a["label"], [pa[i] for i in df_a["id"]]).point
    auc_b = bootstrap_auroc(df_b["label"], [pb[i] for i in df_b["id"]]).point
    # Both must carry a strong UNMATCHED signal -- that is the precondition for
    # the battery tests to mean anything, since a control can only be shown to
    # knock something down if there was something there. The confounded set
    # tops out below the real-signal set because a third of its negatives are
    # gc twins by construction, which is the same overlap that lets matching
    # neutralise gc later.
    assert auc_a > 0.9, auc_a
    assert auc_b > 0.8, auc_b
    assert auc_a > auc_b


def test_battery_counts_every_declared_confound():
    # The source computed its pass flag over a hardcoded subset of confound
    # names, so a confound added to the spec was scored, printed, and then left
    # out of the decision it existed to inform.
    df, probs = _frame(confound_drives=True)
    anchor = bootstrap_auroc(df["label"], [probs[i] for i in df["id"]]).point
    results = run_battery(df, probs, {"other": ["other"], "gc": ["gc"]}, anchor)
    assert set(results) == {"other", "gc"}
    assert results["gc"].verdict == DRIVEN
    assert not battery_passes(results), "a failing confound must sink the battery"


def test_an_empty_spec_or_result_set_raises_rather_than_passing():
    df, probs = _frame()
    with pytest.raises(ValueError, match="nothing to control for"):
        run_battery(df, probs, {}, 0.9)
    with pytest.raises(ValueError, match="empty battery cannot pass"):
        battery_passes({})


def test_missing_columns_and_missing_probabilities_raise():
    df, probs = _frame()
    with pytest.raises(ValueError, match="columns not in frame"):
        evaluate_confound(df, probs, ["nope"], 0.9)
    partial = {k: v for k, v in list(probs.items())[:10]}
    with pytest.raises(ValueError, match="have no probability"):
        evaluate_confound(df, partial, ["other"], 0.9)


def test_incomplete_match_raises_by_default_and_is_reported_when_allowed():
    # Only 5 negatives for 150 positives.
    df, probs = _frame()
    small = pd.concat([df[df["label"] == 1], df[df["label"] == 0].head(5)])
    anchor = 0.9
    with pytest.raises(ValueError, match="matched only"):
        evaluate_confound(small, probs, ["other"], anchor)
    res = evaluate_confound(
        small, probs, ["other"], anchor,
        require_complete_match=False, require_selective_match=False
    )
    assert not res.match_complete
    assert res.n_matched_negatives == 5


def test_format_battery_flags_an_incomplete_match_in_its_text():
    df, probs = _frame()
    small = pd.concat([df[df["label"] == 1], df[df["label"] == 0].head(5)])
    results = {
        "other": evaluate_confound(
            small, probs, ["other"], 0.9, name="other",
            require_complete_match=False, require_selective_match=False
        )
    }
    assert "INCOMPLETE MATCH" in format_battery(results, 0.9)


def test_anchor_at_or_below_chance_is_refused():
    # (auroc - 0.5) / (anchor - 0.5) with anchor <= 0.5 is a non-quantity, not
    # a large number -- the denominator has no above-chance signal in it.
    with pytest.raises(ValueError, match="anchor must be above chance"):
        recovery(0.8, 0.5)
    with pytest.raises(ValueError, match="anchor must be above chance"):
        recovery(0.8, 0.3)
    assert recovery(0.8, 0.7) == pytest.approx(1.5)


def test_bootstrap_refuses_a_single_class_and_reports_a_real_interval():
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_auroc(np.ones(10), np.arange(10.0))
    ci = bootstrap_auroc(
        np.r_[np.ones(50), np.zeros(50)], np.r_[np.ones(50), np.zeros(50)]
    )
    assert ci.point == 1.0 and ci.lo <= ci.hi and ci.excludes_chance


def test_a_wide_interval_is_inconclusive_rather_than_confound_driven():
    # The source had no way to say "too wide to decide" and folded such cases
    # into confound-driven -- reporting an absent measurement as a finding.
    from confound_controls import AurocCI

    wide = AurocCI(point=0.75, lo=0.30, hi=0.98)
    assert verdict(wide, anchor=0.9) == DRIVEN  # opt-in is off by default
    assert verdict(wide, anchor=0.9, max_ci_width=0.3) == INCONCLUSIVE


def test_an_equal_sized_pool_is_refused_as_a_vacuous_control():
    """The failure this library exists to prevent.

    With as many negatives as positives, 1:1 matching without replacement takes
    the WHOLE pool whatever the confound values are. The matched comparison is
    then identical to the unmatched one, so a confound-driven signal comes back
    `confound-robust` -- the control reports the confound ruled out precisely
    when it did no work. Found by this suite: the first version of the
    confound-driven test used 150 vs 150 and returned AUROC 0.9312.
    """
    df, probs = _frame(n_pos=150, n_neg=150, confound_drives=True)
    anchor = bootstrap_auroc(df["label"], [probs[i] for i in df["id"]]).point
    with pytest.raises(ValueError, match="vacuous"):
        evaluate_confound(df, probs, ["gc"], anchor, name="gc")

    # Opting out reproduces the original silent behaviour, and it is WRONG in
    # exactly the way described: a pure-GC signal survives GC matching.
    res = evaluate_confound(df, probs, ["gc"], anchor, name="gc",
                            require_selective_match=False)
    assert not res.match_selective
    assert res.verdict == ROBUST, (
        "if this is no longer ROBUST the vacuity demonstration has changed")
    assert "VACUOUS CONTROL" in format_battery({"gc": res}, anchor)

    # Positive control: the SAME confound with a pool that permits selection
    # reaches the opposite, correct verdict.
    big_df, big_probs = _frame(n_pos=150, n_neg=600, confound_drives=True)
    big_anchor = bootstrap_auroc(
        big_df["label"], [big_probs[i] for i in big_df["id"]]).point
    good = evaluate_confound(big_df, big_probs, ["gc"], big_anchor, name="gc")
    assert good.match_selective and good.verdict == DRIVEN
