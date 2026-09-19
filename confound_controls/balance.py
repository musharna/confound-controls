"""Did the match balance the confound, or only pair things up?

`MatchResult.complete` and `.selective` answer "did every positive get its own
negative" and "did matching choose". Neither asks the question matching exists
to answer. Greedy 1:1 matching without replacement spends the negatives that
sit near the positives first; whoever is left is paired with whatever is left.
The result is a complete, selective match that is still confounded -- and a
classifier that still separates it is then reported as having survived the
control. On a classifier that was NOTHING BUT the confound, 0.3.2 returned
"confound-robust".

The standardized mean difference, with the variance ratio for continuous
confounds, is the usual check on a matched sample (Austin 2009,
doi:10.1002/sim.3697): the difference in group means in units of a standard
deviation, which unlike a t-test does not shrink just because matching threw
samples away. |SMD| < 0.1 is the conventional line for "balanced"; it is a
convention, and `max_smd` is a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DENOMINATORS = ("pooled", "positives")


def _as_matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D (n_samples, n_confounds) array, got {a.ndim}-D")
    return a


def _is_binary(*columns: np.ndarray) -> bool:
    return bool(np.isin(np.concatenate(columns), (0.0, 1.0)).all())


def _variance(col: np.ndarray, binary: bool) -> float:
    """Variance of one group's column. A 0/1 column uses p(1-p): its spread is a
    function of its mean, and the sample estimate only adds noise to that."""
    if binary:
        p = float(col.mean())
        return p * (1.0 - p)
    return float(col.var(ddof=1)) if len(col) > 1 else 0.0


def standardized_mean_differences(pos, neg, *, denominator: str = "pooled", reference=None) -> list:
    """Per-confound (mean_pos - mean_neg) / sd.

    Args:
        pos, neg: (n, n_confounds) confound values of the two groups compared.
        denominator: "pooled" -- sqrt((var_pos + var_neg) / 2), symmetric in the
            groups; or "positives" -- the positives' SD, which is
            what MatchIt's `summary()` reports for 1:1 matching to a fixed group.
        reference: optional (pos, neg) pair the SD is taken FROM. Pass the
            unmatched groups when measuring a matched sample, so that before and
            after share one yardstick; otherwise matching can "improve" an SMD
            merely by changing the spread it is divided by.

    A confound with no spread gives 0.0 when the means agree and inf when they
    do not: the groups differ by something no amount of standardising can scale.
    """
    if denominator not in _DENOMINATORS:
        raise ValueError(f"denominator must be one of {_DENOMINATORS}, got {denominator!r}")
    pos, neg = _as_matrix(pos), _as_matrix(neg)
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(f"confound count differs: {pos.shape[1]} vs {neg.shape[1]}")
    if not len(pos) or not len(neg):
        raise ValueError(f"need both groups; got {len(pos)} positives and {len(neg)} negatives")
    ref_pos, ref_neg = (pos, neg) if reference is None else map(_as_matrix, reference)
    out = []
    for j in range(pos.shape[1]):
        binary = _is_binary(ref_pos[:, j], ref_neg[:, j])
        v_pos = _variance(ref_pos[:, j], binary)
        v_neg = _variance(ref_neg[:, j], binary)
        sd = np.sqrt((v_pos + v_neg) / 2.0) if denominator == "pooled" else np.sqrt(v_pos)
        diff = float(pos[:, j].mean() - neg[:, j].mean())
        if sd == 0:
            out.append(0.0 if diff == 0 else float(np.copysign(np.inf, diff)))
        else:
            out.append(diff / float(sd))
    return out


@dataclass(frozen=True)
class BalanceReport:
    """Confound balance before and after matching, one entry per column."""

    columns: list
    smd_before: list
    smd_after: list
    variance_ratio_after: list
    n_positives_after: int
    n_negatives_after: int
    denominator: str = "pooled"

    @property
    def max_abs_smd_before(self) -> float:
        return float(np.max(np.abs(self.smd_before)))

    @property
    def max_abs_smd_after(self) -> float:
        return float(np.max(np.abs(self.smd_after)))

    @property
    def worst_column(self):
        return self.columns[int(np.argmax(np.abs(self.smd_after)))]

    def balanced(self, threshold: float = 0.1) -> bool:
        return self.max_abs_smd_after <= threshold

    def require_balanced(self, threshold: float = 0.1, *, name: str = "match") -> BalanceReport:
        if not self.balanced(threshold):
            raise ValueError(
                f"{name}: matching did not balance '{self.worst_column}': "
                f"|SMD| = {self.max_abs_smd_after:.3f} after matching "
                f"(was {self.max_abs_smd_before:.3f}) exceeds {threshold}. The matched "
                f"set is still confounded, so a classifier that separates it has not "
                f"been shown to be robust to this confound. Pass caliper=... to drop "
                f"positives with no close negative (try 0.2), supply a larger negative "
                f"pool, or pass max_smd=None to inspect the result anyway."
            )
        return self

    def format(self) -> str:
        width = max(len(str(c)) for c in self.columns)
        lines = [f"  {'confound':{width}s}  SMD before  SMD after  var ratio"]
        for c, b, a, v in zip(
            self.columns, self.smd_before, self.smd_after, self.variance_ratio_after
        ):
            ratio = "      -" if np.isnan(v) else f"{v:7.3f}"
            lines.append(f"  {c!s:{width}s}  {b:10.3f}  {a:9.3f}  {ratio}")
        return "\n".join(lines)


def balance_report(
    pos, neg, matched_pos, matched_neg, *, columns=None, denominator: str = "pooled"
) -> BalanceReport:
    """Balance of `matched_pos` vs `matched_neg`, against the unmatched `pos` vs `neg`.

    Both SMDs are standardised by the UNMATCHED groups' spread. The variance
    ratio (positives / negatives, after matching) is nan for a 0/1 column, where
    it is a restatement of the mean difference and reads as extra evidence.
    """
    names = list(columns) if columns is not None else list(getattr(pos, "columns", []))
    pos, neg = _as_matrix(pos), _as_matrix(neg)
    matched_pos, matched_neg = _as_matrix(matched_pos), _as_matrix(matched_neg)
    if not names:
        names = [f"x{j}" for j in range(pos.shape[1])]
    if len(names) != pos.shape[1]:
        raise ValueError(f"{len(names)} column names for {pos.shape[1]} confounds")
    ratios = []
    for j in range(pos.shape[1]):
        if _is_binary(pos[:, j], neg[:, j]):
            ratios.append(float("nan"))
            continue
        v_neg = _variance(matched_neg[:, j], False)
        v_pos = _variance(matched_pos[:, j], False)
        ratios.append(v_pos / v_neg if v_neg > 0 else float("inf" if v_pos > 0 else "nan"))
    return BalanceReport(
        columns=names,
        smd_before=standardized_mean_differences(pos, neg, denominator=denominator),
        smd_after=standardized_mean_differences(
            matched_pos, matched_neg, denominator=denominator, reference=(pos, neg)
        ),
        variance_ratio_after=ratios,
        n_positives_after=len(matched_pos),
        n_negatives_after=len(matched_neg),
        denominator=denominator,
    )
