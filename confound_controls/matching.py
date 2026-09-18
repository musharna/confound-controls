"""Match negatives to positives on confounds, and say so when you cannot.

Extracted from an internal matched-negatives script (also present
byte-identical in a second internal project -- the code was copied between two
independent repos, which is what marked it as worth extracting).

The matching itself is unchanged. What changed is that it now reports whether
it achieved the 1:1 match it advertises. The original looped

    while True:
        ...
        if pick is not None: ...; break
        if k >= len(neg_ids): break     # <- silently gives up on this positive
        k *= 2

so once the negative pool was exhausted it returned FEWER negatives than
positives and said nothing. A caller comparing AUROC before and after matching
could not tell a clean 1:1 design from one that quietly dropped a third of its
positives -- and the drop is not random, it hits the positives in the densest
region of confound space, exactly where matching matters most.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a matching attempt.

    `complete` is the question a caller actually needs answered: did every
    positive get its own negative? `unmatched_positions` gives the indices of
    the positives that did not, so the caller can inspect rather than guess.
    """

    matched_ids: list
    unmatched_positions: list
    n_positives: int
    n_pool: int = 0
    # Parallel to `matched_ids`: which positive each negative was matched to, and
    # how far apart they are in the space that was matched on (standardised
    # confounds, or the propensity logit). Without these a caller cannot rebuild
    # the pairs, and so cannot check what the match achieved.
    matched_positions: list = field(default_factory=list)
    distances: list = field(default_factory=list)
    # The caliper actually applied, in the units of `distances`; None = none.
    caliper: float | None = None

    @property
    def complete(self) -> bool:
        return not self.unmatched_positions

    @property
    def n_matched(self) -> int:
        return len(self.matched_ids)

    @property
    def selective(self) -> bool:
        """Did matching actually CHOOSE, or take the whole pool?

        1:1 matching without replacement from a pool the same size as the
        positive set consumes every negative, whatever the confound values are.
        The "matched" comparison is then byte-identical to the unmatched one,
        so the control is vacuous -- and a vacuous control does not report
        "inconclusive", it reports the original result, which reads as the
        confound having been ruled out.
        """
        return self.n_matched < self.n_pool

    def require_complete(self) -> MatchResult:
        if not self.complete:
            raise ValueError(
                f"matched only {self.n_matched} of {self.n_positives} positives; "
                f"the negative pool ran out. Unmatched positive positions: "
                f"{self.unmatched_positions[:10]}"
                f"{'...' if len(self.unmatched_positions) > 10 else ''}"
            )
        return self

    def require_selective(self) -> MatchResult:
        if not self.selective:
            raise ValueError(
                f"matching consumed the entire negative pool "
                f"({self.n_matched} matched from {self.n_pool} available), so it "
                f"selected nothing and the control is vacuous -- the matched "
                f"comparison equals the unmatched one and will report the "
                f"confound as ruled out. Supply more negatives than positives, "
                f"or pass require_selective_match=False if you intend this."
            )
        return self


METHODS = ("nearest", "propensity")
_NEWTON_MAX_ITER = 100
# |logit| this large is a fitted probability within 1e-13 of 0 or 1: separation.
_LOGIT_LIMIT = 30.0


def propensity_logit(pos_vecs, neg_vecs) -> tuple[np.ndarray, np.ndarray]:
    """Logit of P(positive | confounds) for each positive and each negative.

    An unpenalised logistic regression fitted by Newton's method, i.e. what R's
    `glm(..., binomial)` fits (it agrees with MatchIt's to 1e-7 on the lalonde
    data). Not scikit-learn's: its default L2 penalty shrinks the coefficients by
    an amount that depends on the units the confounds happen to be in. The logit
    rather than the probability because it is roughly normal, so a distance (and
    a caliper) means the same thing in the tails as in the middle.

    Raises if the confounds separate the classes perfectly. No propensity score
    exists then -- the likelihood has no maximum -- and no negative resembles
    any positive, which is an answer about the data and not something to match
    through.
    """
    pos_vecs = np.asarray(pos_vecs, dtype=float)
    neg_vecs = np.asarray(neg_vecs, dtype=float)
    x = np.vstack([pos_vecs, neg_vecs])
    y = np.r_[np.ones(len(pos_vecs)), np.zeros(len(neg_vecs))]
    if not len(pos_vecs) or not len(neg_vecs):
        raise ValueError("a propensity model needs both positives and negatives")
    mu, sd = x.mean(0), x.std(0)
    keep = sd > 0  # a constant column carries no information and is collinear with the intercept
    design = np.column_stack([np.ones(len(x)), (x[:, keep] - mu[keep]) / sd[keep]])
    beta = np.zeros(design.shape[1])
    for _ in range(_NEWTON_MAX_ITER):
        eta = design @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)
        hessian = design.T @ (design * w[:, None])
        try:
            step = np.linalg.solve(hessian, design.T @ (y - p))
        except np.linalg.LinAlgError:
            step = None
        if step is None or not np.isfinite(step).all() or np.abs(eta).max() > _LOGIT_LIMIT:
            break
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            logit = design @ beta
            return logit[: len(pos_vecs)], logit[len(pos_vecs) :]
    raise ValueError(
        "the propensity model did not converge: the confounds separate positives "
        "from negatives (almost) perfectly, or are collinear. With no overlap "
        "there is no negative that resembles a positive on these confounds, so "
        "matching cannot control for them."
    )


def match_negatives(
    pos_vecs, neg_ids, neg_vecs, *, method: str = "nearest", caliper: float | None = None
) -> MatchResult:
    """Greedy 1:1 nearest-neighbour match without replacement.

    `method="nearest"` measures distance on the standardised confound columns
    (standardised over positives and negatives jointly, so a confound on a wide
    scale does not dominate through its units). `method="propensity"` collapses
    the confounds to one number first -- the logit of the propensity score --
    which is what keeps matching workable once there are many confounds and
    every sample is far from every other. Positives are then taken in order of
    decreasing propensity: the hardest to match choose first, while the
    negatives that could serve them are still available.

    A `caliper` refuses a pairing further apart than that many standard
    deviations -- of the standardised columns (i.e. raw units of distance) for
    "nearest", of the propensity logit for "propensity", where 0.2 standard
    deviations of the logit is the recommended width (Austin 2011,
    doi:10.1002/pst.433). A positive with no unused
    negative inside the caliper is left unmatched and reported, not paired with
    something that does not resemble it.

    Args:
        pos_vecs: (n_pos, n_confounds) confound values for the positives.
        neg_ids: identifiers for the negatives, parallel to `neg_vecs`.
        neg_vecs: (n_neg, n_confounds) confound values for the negatives.

    Returns:
        MatchResult -- inspect `.complete` before trusting a downstream metric,
        or call `.require_complete()` to make an incomplete match an error.
    """
    pos_vecs = np.asarray(pos_vecs, dtype=float)
    neg_vecs = np.asarray(neg_vecs, dtype=float)
    neg_ids = list(neg_ids)

    if pos_vecs.ndim != 2 or neg_vecs.ndim != 2:
        raise ValueError("pos_vecs and neg_vecs must both be 2-D")
    if pos_vecs.shape[1] != neg_vecs.shape[1]:
        raise ValueError(
            f"confound count differs: positives have {pos_vecs.shape[1]}, "
            f"negatives have {neg_vecs.shape[1]}"
        )
    if len(neg_ids) != neg_vecs.shape[0]:
        raise ValueError(
            f"neg_ids has {len(neg_ids)} entries but neg_vecs has {neg_vecs.shape[0]} rows"
        )
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if caliper is not None and not caliper > 0:
        raise ValueError(f"caliper must be positive, got {caliper!r}")
    if not len(neg_ids):
        return MatchResult([], list(range(len(pos_vecs))), len(pos_vecs), 0)

    order = range(len(pos_vecs))
    limit = caliper
    if method == "propensity":
        if not len(pos_vecs):
            return MatchResult([], [], 0, len(neg_ids))
        pos_logit, neg_logit = propensity_logit(pos_vecs, neg_vecs)
        pos_s, neg_s = pos_logit.reshape(-1, 1), neg_logit.reshape(-1, 1)
        order = np.argsort(-pos_logit, kind="stable")
        if caliper is not None:
            # In SDs of the logit over every sample, positives and negatives
            # together -- the scale MatchIt's `std.caliper` uses.
            limit = float(caliper * np.r_[pos_logit, neg_logit].std(ddof=1))
    else:
        allv = np.vstack([pos_vecs, neg_vecs])
        mu, sd = allv.mean(0), allv.std(0)
        # A confound that is constant across every sample carries no information;
        # dividing by its zero spread would produce nan distances and match at
        # random, which looks exactly like a successful match.
        sd = np.where(sd == 0, 1.0, sd)
        pos_s = (pos_vecs - mu) / sd
        neg_s = (neg_vecs - mu) / sd

    tree = cKDTree(neg_s)
    used: set[int] = set()
    picks: dict[int, tuple[int, float]] = {}

    for position in order:
        p = pos_s[position]
        k = 1
        pick = None
        while True:
            k = min(k, len(neg_ids))
            dists, idxs = tree.query(p, k=k)
            dists, idxs = np.atleast_1d(dists), np.atleast_1d(idxs)
            pick = next(((int(i), float(d)) for d, i in zip(dists, idxs) if i not in used), None)
            # Neighbours come back nearest first, so once the furthest one seen
            # is outside the caliper no unseen one can be inside it.
            beyond = limit is not None and dists[-1] > limit
            if pick is not None or k >= len(neg_ids) or beyond:
                break
            k *= 2
        if pick is not None and (limit is None or pick[1] <= limit):
            used.add(pick[0])
            picks[int(position)] = pick

    # Report in the positives' own order whatever order they were matched in.
    positions = sorted(picks)
    return MatchResult(
        matched_ids=[neg_ids[picks[i][0]] for i in positions],
        unmatched_positions=[i for i in range(len(pos_s)) if i not in picks],
        n_positives=len(pos_s),
        n_pool=len(neg_ids),
        matched_positions=positions,
        distances=[picks[i][1] for i in positions],
        caliper=limit,
    )
