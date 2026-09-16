"""Sequence-level ablation: knock out a span, keep composition, say whether it worked.

Extracted from an internal project (byte-identical in a second one):
`aim3_shuffle_promoters.dinucl_shuffle`, `ig_knockout_ism.knockout_instance` /
`delta_ci`, and `cg_ism._nonoverlap_start`. `cg_ism`'s own docstring says it
"reuses ig_knockout_ism (knockout_instance, delta_ci)", so the reuse was
already explicit -- it just had nowhere to live.

## What changed: three silent fallbacks, all in the same direction

Each of these returned a plausible value on failure, and each failure makes an
ablation weaker or absent -- which reads as the model SURVIVING it, the most
reassuring answer available.

1. `dinucl_shuffle` ended with

       # deterministic fallback: identity preserves counts (rare; logged by caller)
       return seq

   An un-shuffleable sequence came back UNCHANGED. Downstream that is a
   knockout that knocked nothing out, indistinguishable from a real one.
   "Logged by caller" is a promise, not a mechanism.

2. `_nonoverlap_start` tried 50 times to place a length-matched control span
   clear of the real motif spans, then returned `rng.randint(0, hi)` regardless
   -- a control span that may OVERLAP the motifs, i.e. a partial real knockout
   masquerading as the negative control it is being compared against.

3. `delta_ci` resampled groups with no check that there were enough distinct
   groups for the interval to mean anything.

The dinucleotide-shuffle algorithm itself (Altschul-Erikson) is ported
unchanged, including the determinism fix its author found the hard way -- see
`_vertices`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def dinucleotide_counts(seq: str) -> dict:
    c: dict = defaultdict(int)
    for i in range(len(seq) - 1):
        c[seq[i : i + 2]] += 1
    return dict(c)


def _vertices(graph):
    """`list(graph)`, never `set(graph)`.

    dict key order is insertion order since 3.7, so it is derived from the
    sequence. Set iteration order depends on the per-process randomized hash of
    the string keys, which made `rng.choice`/`rng.shuffle` consume the PRNG in a
    different order across process invocations -- so the same `(seq, seed)`
    produced DIFFERENT shuffles run to run despite the PRNG being correctly
    seeded. Preserved verbatim from the source, which carries the same warning.
    """
    return list(graph)


def _connected_to_last(last, last_edges, verts) -> bool:
    """Altschul-Erikson connectivity: every vertex with out-edges reaches `last`."""
    intree = {v: False for v in verts}
    intree[last] = True
    changed = True
    while changed:
        changed = False
        for v in verts:
            if not intree[v] and v in last_edges and intree.get(last_edges[v], False):
                intree[v] = True
                changed = True
    return all(intree[v] for v in verts)


@dataclass(frozen=True)
class ShuffleResult:
    """A shuffle attempt, and whether it actually changed anything.

    `changed` is the question the source could not answer. A shuffle that
    returns its input is not a shuffle, and every ablation built on it is inert.
    """

    sequence: str
    changed: bool
    attempts: int

    def require_changed(self) -> ShuffleResult:
        if not self.changed:
            raise ValueError(
                f"dinucleotide shuffle returned the input unchanged after "
                f"{self.attempts} attempts, so any ablation built on it is "
                f"inert and will report the model as surviving an ablation "
                f"that never happened. Short, low-complexity or homopolymeric "
                f"sequences often cannot be reshuffled at all."
            )
        return self


def dinucleotide_shuffle(seq: str, seed: int, attempts: int = 50) -> ShuffleResult:
    """Altschul-Erikson shuffle preserving exact dinucleotide counts.

    Composition and dinucleotide frequency are held fixed while order is
    destroyed, which is what makes it a control for "did the model read order,
    or just composition?".

    Returns a ShuffleResult -- check `.changed` (or call `.require_changed()`)
    before using the output as an ablation.
    """
    if len(seq) < 3:
        return ShuffleResult(seq, False, 0)

    rng = random.Random(seed)
    last = seq[-1]
    graph = defaultdict(list)
    for i in range(len(seq) - 1):
        graph[seq[i]].append(seq[i + 1])
    verts = _vertices(graph)

    for attempt in range(attempts):
        last_edges = {}
        ok = True
        for v in verts:
            if v == last:
                continue
            if not graph[v]:
                ok = False
                break
            last_edges[v] = rng.choice(graph[v])
        if not ok or not _connected_to_last(last, last_edges, verts):
            rng = random.Random(seed + 7919 * (attempt + 1))
            continue

        edges = {v: list(succ) for v, succ in graph.items()}
        for v, w in last_edges.items():
            edges[v].remove(w)
            rng.shuffle(edges[v])
            edges[v].append(w)  # the chosen last-edge departs last
        for v, succ_edges in edges.items():
            if v not in last_edges:
                rng.shuffle(succ_edges)

        result = [seq[0]]
        cur = seq[0]
        for _ in range(len(seq) - 1):
            if not edges.get(cur):
                ok = False
                break
            nxt = edges[cur].pop(0)
            result.append(nxt)
            cur = nxt
        out = "".join(result)
        if ok and dinucleotide_counts(out) == dinucleotide_counts(seq):
            return ShuffleResult(out, out != seq, attempt + 1)
        rng = random.Random(seed + 7919 * (attempt + 1))

    # The source returned `seq` here silently. It is still the only
    # count-preserving answer available -- but it is reported, not disguised.
    #
    # NOT COVERED BY THE SUITE, and deliberately marked so rather than left
    # looking tested: reaching this line needs 50 consecutive attempts to fail
    # connectivity or the count check, and a probe over 8 sequences x 30 seeds
    # (homopolymers, dinucleotide repeats, 2-4mers included) reached it zero
    # times. The unshuffleable cases users actually hit -- "AAAAAAAA", or a
    # 12-mer under an unlucky seed -- exit through the success path above with
    # `out == seq`, which IS covered. A mutation planted on this line survives
    # the suite; that is a true coverage gap on an unreachable backstop, not a
    # missing test.
    return ShuffleResult(seq, False, attempts)


def knockout_span(seq: str, start: int, end: int, seed: int) -> ShuffleResult:
    """Scramble `seq[start:end]` in place; flanks stay byte-identical."""
    if not 0 <= start < end <= len(seq):
        raise ValueError(f"span [{start}, {end}) is not inside a sequence of length {len(seq)}")
    inner = dinucleotide_shuffle(seq[start:end], seed)
    return ShuffleResult(seq[:start] + inner.sequence + seq[end:], inner.changed, inner.attempts)


@dataclass(frozen=True)
class ControlSpan:
    start: int
    length: int
    disjoint: bool

    @property
    def end(self) -> int:
        return self.start + self.length

    def require_disjoint(self) -> ControlSpan:
        if not self.disjoint:
            raise ValueError(
                f"could not place a length-{self.length} control span clear of "
                f"the real spans; the returned start {self.start} OVERLAPS them, "
                f"so the 'random control' is a partial real knockout and will be "
                f"compared against itself. Use a longer sequence, a shorter "
                f"total, or accept it explicitly."
            )
        return self


def _draw_start(rng, lo: int, hi: int) -> int:
    """Draw an integer in [lo, hi] INCLUSIVE, whichever PRNG flavour was passed.

    The three PRNGs this package already uses disagree about the upper bound:
    numpy's RandomState.randint and Generator.integers EXCLUDE it, stdlib
    random.Random.randint INCLUDES it -- and Generator has no `.randint` at all.
    `rng.randint(0, hi)` therefore meant a different range for different callers
    and, under numpy, could never draw the last legal start.
    """
    if hasattr(rng, "integers"):  # numpy Generator
        return int(rng.integers(lo, hi + 1))
    if hasattr(rng, "random_sample"):  # numpy RandomState
        return int(rng.randint(lo, hi + 1))
    return int(rng.randint(lo, hi))  # stdlib random.Random


def sample_control_span(length: int, total: int, spans, rng, attempts: int = 50) -> ControlSpan:
    """Place a length-matched control span avoiding every real span.

    The source returned a random start after 50 failures with no indication,
    so a control that overlapped the motifs it was controlling for looked
    identical to one that did not.
    """
    if total <= 0:
        raise ValueError(f"control span length must be positive, got {total}")
    if total > length:
        raise ValueError(
            f"control span length {total} is longer than the sequence ({length}), "
            f"so no placement fits; the source clamped the range and returned a "
            f"span running off the end, flagged as if it were valid."
        )
    spans = list(spans)
    max_start = length - total  # INCLUSIVE: a span may sit flush at the end

    def _clear(st: int) -> bool:
        return all(st + total <= s or st >= e for s, e in spans)

    for _ in range(attempts):
        st = _draw_start(rng, 0, max_start)
        if _clear(st):
            return ControlSpan(st, total, True)
    # Report what the fallback draw ACTUALLY is. The source hardcoded False, so a
    # span that happened to land clear was reported as overlapping, and callers
    # discarded a usable control (or trusted require_disjoint's error over the data).
    st = _draw_start(rng, 0, max_start)
    return ControlSpan(st, total, _clear(st))


@dataclass(frozen=True)
class GroupedDelta:
    mean: float
    lo: float
    hi: float
    n_groups: int

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


def grouped_delta_ci(
    real, ko, groups, n: int = 2000, seed: int = 42, min_groups: int = 5
) -> GroupedDelta:
    """Paired CLUSTER bootstrap on mean(real - ko), resampling whole groups.

    Resampling groups rather than rows is the point: promoters from one paralog
    family are not independent observations, and a row-level bootstrap would
    report an interval far narrower than the data supports.

    `min_groups` guards the case the source did not check -- with a handful of
    clusters the percentile interval is a number, not evidence.
    """
    real = np.asarray(real, dtype=float)
    ko = np.asarray(ko, dtype=float)
    groups = np.asarray(groups)
    if not (real.shape == ko.shape == groups.shape):
        raise ValueError(f"shapes differ: real={real.shape}, ko={ko.shape}, groups={groups.shape}")
    if real.size == 0:
        raise ValueError("no observations")

    d = real - ko
    uniq = np.unique(groups)
    if len(uniq) < min_groups:
        raise ValueError(
            f"only {len(uniq)} distinct group(s); a cluster bootstrap resamples "
            f"groups, so the interval would be drawn from at most {len(uniq)} "
            f"distinct values. Pass min_groups=1 to override if you know what "
            f"that interval means."
        )

    rng = np.random.RandomState(seed)
    index_of = {g: np.where(groups == g)[0] for g in uniq}
    stats = []
    for _ in range(n):
        gs = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in gs])
        stats.append(float(d[idx].mean()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return GroupedDelta(float(d.mean()), float(lo), float(hi), len(uniq))


CONFIRMED = "confirmed"
NOT_CONFIRMED = "not-confirmed"


def confirm_knockout(pooled: GroupedDelta, control: GroupedDelta) -> str:
    """The source's confirmation criterion, unchanged.

    Confirmed iff the pooled knockout drops the score, its interval clears
    zero, AND it drops more than a length-matched control knockout does. The
    third clause is what stops "any perturbation hurts" being read as "this
    element matters".
    """
    return (
        CONFIRMED
        if (pooled.mean > 0 and pooled.lo > 0 and pooled.mean > control.mean)
        else NOT_CONFIRMED
    )
