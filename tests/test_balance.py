"""Balance diagnostics, calipers and propensity matching.

The defect these exist for: 0.3.2 reported a match as `complete` and
`selective` and never asked whether it had BALANCED anything. Greedy 1:1
matching without replacement runs out of nearby negatives; the remaining
positives are paired with whatever is left. The matched set is still confounded,
the classifier still separates it, and the battery reads that as the signal
having survived the control -- "confound-robust" -- for a classifier that is
nothing but the confound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from confound_controls import (
    DRIVEN,
    balance_report,
    evaluate_confound,
    format_battery,
    match_negatives,
    propensity_logit,
    run_battery,
    standardized_mean_differences,
)

DATA = Path(__file__).parent / "data"
COVARIATES = ["age", "educ", "married", "nodegree", "re74", "re75"]

# summary(matchit(...)) from MatchIt 4.5.5, copied from the R console.
MATCHIT_SMD_ALL = [-0.309445, 0.054965, -0.826309, 0.244970, -0.721084, -0.290263]
MATCHIT_SMD_MATCHED = [0.037219, -0.116229, 0.000000, 0.095634, -0.070019, -0.003952]
MATCHIT_VARRATIO_ALL = {"age": 0.439995, "educ": 0.495893, "re74": 0.518128, "re75": 0.956293}
MATCHIT_CALIPER = 0.2015685002


@pytest.fixture(scope="module")
def lalonde():
    df = pd.read_csv(DATA / "lalonde.csv")
    return df[df.treat == 1], df[df.treat == 0], df


def confounded_frame(n_pos=300, n_neg=900, shift=2.0, seed=0):
    rng = np.random.default_rng(seed)
    gc = np.r_[rng.normal(shift, 1, n_pos), rng.normal(0, 1, n_neg)]
    df = pd.DataFrame(
        {
            "id": [f"s{i}" for i in range(n_pos + n_neg)],
            "label": [1] * n_pos + [0] * n_neg,
            "gc": gc,
        }
    )
    prob = dict(zip(df.id, 1 / (1 + np.exp(-gc))))  # the classifier IS the confound
    y = df.label.to_numpy()
    p = np.array([prob[i] for i in df.id])
    from sklearn.metrics import roc_auc_score

    return df, prob, float(roc_auc_score(y, p))


def test_smd_and_variance_ratio_agree_with_matchit(lalonde):
    pos, neg, _ = lalonde
    smd = standardized_mean_differences(pos[COVARIATES], neg[COVARIATES], denominator="positives")
    assert smd == pytest.approx(MATCHIT_SMD_ALL, abs=5e-6)
    report = balance_report(pos[COVARIATES], neg[COVARIATES], pos[COVARIATES], neg[COVARIATES])
    for col, want in MATCHIT_VARRATIO_ALL.items():
        assert report.variance_ratio_after[COVARIATES.index(col)] == pytest.approx(want, abs=5e-6)
    # A variance ratio is not defined for a 0/1 column; reporting one invites reading it.
    assert np.isnan(report.variance_ratio_after[COVARIATES.index("married")])


def test_pooled_denominator_is_the_mean_of_the_two_variances():
    pos = np.array([[0.0], [2.0], [4.0]])  # var 4
    neg = np.array([[0.0], [1.0], [2.0]])  # var 1, mean diff 1
    assert standardized_mean_differences(pos, neg) == pytest.approx([1 / np.sqrt(2.5)])
    assert standardized_mean_differences(pos, neg, denominator="positives") == pytest.approx([0.5])


def test_constant_column_is_zero_when_equal_and_infinite_when_not():
    same = standardized_mean_differences(np.ones((3, 1)), np.ones((4, 1)))
    apart = standardized_mean_differences(np.ones((3, 1)), np.zeros((4, 1)))
    assert same == [0.0]
    assert apart == [np.inf]


def test_propensity_logit_agrees_with_matchit_glm(lalonde):
    pos, neg, _ = lalonde
    ref = pd.read_csv(DATA / "lalonde_matchit_logit.csv").set_index("id").logit
    pos_logit, neg_logit = propensity_logit(pos[COVARIATES], neg[COVARIATES])
    assert np.max(np.abs(pos_logit - ref.loc[pos.id].to_numpy())) < 1e-6
    assert np.max(np.abs(neg_logit - ref.loc[neg.id].to_numpy())) < 1e-6


def test_perfectly_separated_confound_has_no_propensity_score():
    with pytest.raises(ValueError, match="did not converge: the confounds separate"):
        propensity_logit(np.array([[5.0], [6.0], [7.0]]), np.array([[0.0], [1.0], [2.0]]))
    # Control: overlapping groups fit, and positives score higher on average.
    rng = np.random.default_rng(1)
    pos_logit, neg_logit = propensity_logit(rng.normal(1, 1, (60, 2)), rng.normal(0, 1, (90, 2)))
    assert pos_logit.mean() > neg_logit.mean()


def test_propensity_caliper_match_reproduces_matchit_pairs(lalonde):
    pos, neg, _ = lalonde
    ref = pd.read_csv(DATA / "lalonde_matchit_pairs.csv").dropna()
    _, neg_logit = propensity_logit(pos[COVARIATES], neg[COVARIATES])
    result = match_negatives(
        pos[COVARIATES].to_numpy(), neg.id.tolist(), neg[COVARIATES].to_numpy(),
        method="propensity", caliper=0.2,
    )  # fmt: skip
    assert result.caliper == pytest.approx(MATCHIT_CALIPER, rel=1e-7)
    assert (result.n_matched, len(result.unmatched_positions)) == (184, 1)
    ours = {pos.id.iloc[i]: c for i, c in zip(result.matched_positions, result.matched_ids)}
    theirs = dict(zip(ref.treated, ref.control))
    assert set(ours) == set(theirs)
    # lalonde has duplicated rows, so several controls share a logit exactly and
    # either is "the nearest". Every pair is MatchIt's, or a tie with MatchIt's.
    logit_of = dict(zip(neg.id, neg_logit))
    differing = [t for t in theirs if ours[t] != theirs[t]]
    assert len(differing) <= 10
    for t in differing:
        assert logit_of[ours[t]] == pytest.approx(logit_of[theirs[t]], abs=1e-9), t
    matched_pos = pos.iloc[result.matched_positions][COVARIATES]
    matched_neg = neg.set_index("id").loc[result.matched_ids][COVARIATES]
    smd = standardized_mean_differences(
        matched_pos, matched_neg, denominator="positives", reference=(pos[COVARIATES], neg[COVARIATES])
    )  # fmt: skip
    assert smd == pytest.approx(MATCHIT_SMD_MATCHED, abs=5e-6)


def test_caliper_leaves_far_positives_unmatched_and_reports_them():
    pos = np.array([[0.0], [0.1], [9.0]])
    neg = np.array([[0.05], [0.2], [0.3], [0.4]])
    loose = match_negatives(pos, list("abcd"), neg)
    assert (loose.complete, loose.matched_positions) == (True, [0, 1, 2])
    tight = match_negatives(pos, list("abcd"), neg, caliper=0.5)
    assert (tight.matched_positions, tight.unmatched_positions) == ([0, 1], [2])
    assert tight.matched_ids == ["a", "b"]
    assert max(tight.distances) <= 0.5
    with pytest.raises(ValueError, match="caliper must be positive"):
        match_negatives(pos, list("abcd"), neg, caliper=0)


def test_classifier_that_is_the_confound_is_not_called_robust():
    df, prob, anchor = confounded_frame()
    with pytest.raises(ValueError, match=r"gc: .*\|SMD\| = 0\.\d+ .*exceeds 0\.1") as err:
        evaluate_confound(df, prob, ["gc"], anchor, name="gc", bootstrap_n=200)
    assert "caliper" in str(err.value)
    # Positive control: the same data, matched within a caliper, IS evaluable --
    # and the verdict is the true one.
    result = evaluate_confound(df, prob, ["gc"], anchor, name="gc", caliper=0.1, bootstrap_n=200)
    assert result.balance.max_abs_smd_after < 0.1
    assert result.verdict == DRIVEN
    assert 0 < result.n_positives < 300
    assert result.n_positives == result.n_matched_negatives
    assert result.n_positives + len(result.unmatched_positions) == 300


def test_imbalance_can_be_inspected_instead_of_raised():
    df, prob, anchor = confounded_frame()
    results = run_battery(df, prob, {"gc": ["gc"]}, anchor, max_smd=None, bootstrap_n=200)
    assert results["gc"].balance.max_abs_smd_after > 0.1
    text = format_battery(results, anchor)
    assert "[IMBALANCED: max |SMD| = 0." in text
    # Control: a balanced battery carries no such flag.
    balanced = run_battery(df, prob, {"gc": ["gc"]}, anchor, caliper=0.1, bootstrap_n=200)
    text = format_battery(balanced, anchor)
    assert "IMBALANCED" not in text
    dropped = len(balanced["gc"].unmatched_positions)
    assert dropped > 0 and f"[INCOMPLETE MATCH: {dropped} positives left out]" in text


def test_incomplete_match_is_scored_as_the_one_to_one_design_it_claims():
    # 0.3.2 scored ALL positives against the fewer matched negatives.
    df, prob, anchor = confounded_frame(n_pos=40, n_neg=30, shift=1.0, seed=3)
    result = evaluate_confound(
        df, prob, ["gc"], anchor, require_complete_match=False,
        require_selective_match=False, max_smd=None, bootstrap_n=100,
    )  # fmt: skip
    assert (result.n_positives, result.n_matched_negatives) == (30, 30)
    assert len(result.unmatched_positions) == 10
