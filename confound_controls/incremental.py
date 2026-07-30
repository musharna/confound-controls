"""Does the new feature add anything beyond the confounds?

Extracted from arf_promoter_analysis/scripts/aim3_incremental_validity.py
(byte-identical in phelipanche-fm). Fit a confound-only model and a
confound+feature model, evaluate both on the same held-out rows, and put a
paired bootstrap interval on the AUROC difference.

Paired is the point. Two independent intervals on two AUROCs overlap far more
often than the interval on their difference excludes zero, because the pair is
computed on the SAME resampled rows and the shared variance cancels. Comparing
two separately-reported AUROCs by eye is the error the paired delta exists to
prevent.

The source's verdict was two-valued:

    verdict = "FM-adds-signal" if lo > 0 else "no-added-signal"

so an interval straddling zero, an interval too wide to say anything, and an
interval lying entirely BELOW zero -- the augmented model actively worse --
all reported as "no-added-signal". The last of those is not an absence of
signal, it is a finding, and it was being discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ADDS = "adds-signal"
HARMS = "harms"
NO_EFFECT = "no-added-signal"
UNDERPOWERED = "underpowered"


@dataclass(frozen=True)
class DeltaResult:
    auroc_base: float
    auroc_augmented: float
    delta: float
    lo: float
    hi: float
    p_delta_le_0: float
    verdict: str
    n_resamples_used: int

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


def paired_delta_auroc(
    y,
    p_base,
    p_augmented,
    n: int = 2000,
    seed: int = 42,
    max_ci_width: float | None = None,
) -> DeltaResult:
    """Bootstrap the AUROC difference on paired resamples.

    `max_ci_width` opts in to reporting UNDERPOWERED rather than NO_EFFECT when
    the interval is too wide to distinguish the two. Off by default, because
    turning it on silently would reclassify existing results.
    """
    y = np.asarray(y)
    p_base = np.asarray(p_base)
    p_augmented = np.asarray(p_augmented)
    if not (len(y) == len(p_base) == len(p_augmented)):
        raise ValueError(
            f"lengths differ: y={len(y)}, base={len(p_base)}, "
            f"augmented={len(p_augmented)}"
        )
    if len(np.unique(y)) < 2:
        raise ValueError("AUROC needs both classes present in y")

    base = roc_auc_score(y, p_base)
    aug = roc_auc_score(y, p_augmented)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        # Same rows for both models -- that is what makes it paired.
        deltas.append(
            roc_auc_score(y[b], p_augmented[b]) - roc_auc_score(y[b], p_base[b])
        )

    if len(deltas) < n // 2:
        raise ValueError(
            f"only {len(deltas)} of {n} resamples contained both classes; the "
            f"interval would rest on too few replicates to mean anything "
            f"(n_pos={int((y == 1).sum())}, n_neg={int((y == 0).sum())})"
        )

    deltas_arr = np.asarray(deltas)
    lo, hi = np.percentile(deltas_arr, [2.5, 97.5])
    p_le0 = float(np.mean(deltas_arr <= 0))

    if lo > 0:
        verdict = ADDS
    elif hi < 0:
        # Not "no signal" -- the augmented model is reliably WORSE, which the
        # source folded into its no-signal branch and never surfaced.
        verdict = HARMS
    elif max_ci_width is not None and (hi - lo) > max_ci_width:
        verdict = UNDERPOWERED
    else:
        verdict = NO_EFFECT

    return DeltaResult(
        auroc_base=float(base),
        auroc_augmented=float(aug),
        delta=float(aug - base),
        lo=float(lo),
        hi=float(hi),
        p_delta_le_0=p_le0,
        verdict=verdict,
        n_resamples_used=len(deltas),
    )


def _fit_predict(X_train, y_train, X_test, seed: int = 42):
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=seed
    )
    model.fit(scaler.transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_test))[:, 1]


def incremental_validity(
    X_train_confounds,
    X_test_confounds,
    feature_train,
    feature_test,
    y_train,
    y_test,
    *,
    seed: int = 42,
    n: int = 2000,
    max_ci_width: float | None = None,
) -> DeltaResult:
    """Confound-only vs confound+feature, compared with a paired delta.

    `feature_train` must be out-of-fold. Fitting the feature on the same rows
    the confound model is evaluated against leaks the label into the augmented
    model and manufactures the very lift this control is testing for -- the
    source computed it with GroupKFold for exactly that reason.
    """
    X_train_confounds = np.asarray(X_train_confounds, dtype=float)
    X_test_confounds = np.asarray(X_test_confounds, dtype=float)
    feature_train = np.asarray(feature_train, dtype=float).reshape(-1, 1)
    feature_test = np.asarray(feature_test, dtype=float).reshape(-1, 1)

    if X_train_confounds.shape[0] != feature_train.shape[0]:
        raise ValueError(
            f"train rows differ: confounds={X_train_confounds.shape[0]}, "
            f"feature={feature_train.shape[0]}"
        )
    if X_test_confounds.shape[0] != feature_test.shape[0]:
        raise ValueError(
            f"test rows differ: confounds={X_test_confounds.shape[0]}, "
            f"feature={feature_test.shape[0]}"
        )

    p_base = _fit_predict(X_train_confounds, y_train, X_test_confounds, seed)
    p_aug = _fit_predict(
        np.column_stack([X_train_confounds, feature_train]),
        y_train,
        np.column_stack([X_test_confounds, feature_test]),
        seed,
    )
    return paired_delta_auroc(
        y_test, p_base, p_aug, n=n, seed=seed, max_ci_width=max_ci_width
    )
