"""AUROC with a bootstrap interval, and the anchored verdict built on it.

Extracted from an internal probe-common module (bootstrap)
and aim3_matched_negatives.py (recovery/verdict). The maths is unchanged; the
study constants are gone.

The original had

    ANCHOR = 0.629

as a module-level constant -- one study's own baseline AUROC, compiled into the
library. Every verdict the module produced was relative to it, and importing
the module anywhere else silently scored against that number. It is now a
required argument, because there is no defensible default for "what does good
look like in your problem".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

# Verdict vocabulary. `INCONCLUSIVE` has no counterpart in the source, which
# had no way to say "this interval is too wide to decide" and folded those
# cases into `confound-driven` -- reporting an absent measurement as a finding.
ROBUST = "confound-robust"
DRIVEN = "confound-driven"
PARTIAL = "partial"
INCONCLUSIVE = "inconclusive"
INVERTED = "inverted"  # the interval sits entirely BELOW chance


@dataclass(frozen=True)
class AurocCI:
    point: float
    lo: float
    hi: float

    @property
    def excludes_chance(self) -> bool:
        return self.lo > 0.5 or self.hi < 0.5


def bootstrap_auroc(y, p, n: int = 2000, seed: int = 42) -> AurocCI:
    """Percentile bootstrap CI for AUROC.

    Resamples that end up single-class are skipped -- AUROC is undefined for
    them. If too many are skipped the interval is not trustworthy, so this
    raises rather than returning a confident-looking number computed from a
    handful of replicates.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"y has {y.shape[0]} entries, p has {p.shape[0]}")
    if len(np.unique(y)) < 2:
        raise ValueError("AUROC needs both classes present in y")

    point = roc_auc_score(y, p)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    stats = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        stats.append(roc_auc_score(y[b], p[b]))

    if len(stats) < n // 2:
        raise ValueError(
            f"only {len(stats)} of {n} bootstrap resamples contained both "
            f"classes; the interval would be built from too few replicates to "
            f"mean anything (n_pos={int((y == 1).sum())}, "
            f"n_neg={int((y == 0).sum())})"
        )
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return AurocCI(float(point), float(lo), float(hi))


def recovery(auroc: float, anchor: float) -> float:
    """How much of the anchor's above-chance signal survived, in [0, 1]-ish.

    `anchor` is the unmatched/uncontrolled AUROC this result is being compared
    against. An anchor at or below chance makes the ratio meaningless rather
    than merely large, so it is refused.
    """
    if anchor <= 0.5:
        raise ValueError(
            f"anchor must be above chance to divide by its signal; got {anchor}. "
            f"An anchor at or below 0.5 has no above-chance signal to recover, "
            f"so the ratio is a non-quantity, not a small number."
        )
    return (auroc - 0.5) / (anchor - 0.5)


def verdict(
    ci: AurocCI,
    anchor: float,
    recovery_threshold: float = 0.70,
    max_ci_width: float | None = None,
) -> str:
    """Classify a controlled result against its anchor.

    - ROBUST: enough of the anchor's signal survived AND the interval clears chance
    - DRIVEN: the interval straddles chance -- the control removed the signal
    - PARTIAL: signal is real but diminished
    - INVERTED: the interval clears chance from BELOW -- the ranking reversed
    - INCONCLUSIVE: the interval is too wide to support any of the above

    `max_ci_width` is opt-in because the source had no such concept, and adding
    it silently would reclassify existing results.
    """
    if max_ci_width is not None and (ci.hi - ci.lo) > max_ci_width:
        return INCONCLUSIVE
    if recovery(ci.point, anchor) >= recovery_threshold and ci.excludes_chance:
        return ROBUST
    if not ci.excludes_chance:
        return DRIVEN
    if ci.hi < 0.5:
        # Excludes chance, but from below: the control did not diminish the
        # signal, it REVERSED it. Calling that "partial" ("signal is real but
        # diminished") reports a pathology as a weak positive.
        return INVERTED
    return PARTIAL
