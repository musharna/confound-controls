"""Match negatives to positives on confounds, and say so when you cannot.

Extracted from arf_promoter_analysis/scripts/aim3_matched_negatives.py (also
present byte-identical in phelipanche-fm -- the code was copied between two
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

from dataclasses import dataclass

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

    def require_complete(self) -> "MatchResult":
        if not self.complete:
            raise ValueError(
                f"matched only {self.n_matched} of {self.n_positives} positives; "
                f"the negative pool ran out. Unmatched positive positions: "
                f"{self.unmatched_positions[:10]}"
                f"{'...' if len(self.unmatched_positions) > 10 else ''}"
            )
        return self

    def require_selective(self) -> "MatchResult":
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


def match_negatives(pos_vecs, neg_ids, neg_vecs) -> MatchResult:
    """Greedy 1:1 nearest-neighbour match without replacement, on standardised columns.

    Columns are standardised over positives and negatives jointly so that a
    confound measured on a wide scale does not dominate the distance purely
    through its units.

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
            f"neg_ids has {len(neg_ids)} entries but neg_vecs has "
            f"{neg_vecs.shape[0]} rows"
        )
    if not len(neg_ids):
        return MatchResult([], list(range(len(pos_vecs))), len(pos_vecs), 0)

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
    matched: list = []
    unmatched: list[int] = []

    for position, p in enumerate(pos_s):
        k = 1
        pick = None
        while True:
            k = min(k, len(neg_ids))
            _, idxs = tree.query(p, k=k)
            idxs = np.atleast_1d(idxs)
            pick = next((int(i) for i in idxs if i not in used), None)
            if pick is not None or k >= len(neg_ids):
                break
            k *= 2
        if pick is None:
            unmatched.append(position)
        else:
            used.add(pick)
            matched.append(neg_ids[pick])

    return MatchResult(matched, unmatched, len(pos_s), len(neg_ids))
