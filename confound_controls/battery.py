"""Run a confound battery: for each confound, match on it and re-score.

Extracted from an internal matched-negatives script, whose
main() hardcoded the study in four separate places:

    CONFOUNDS = os.path.join(REPO, "results/aim3_control/confounds.tsv")
    ANCHOR    = 0.629
    KMER_COLS = [f"kmer3_{i}" for i in range(64)]
    CONFOUNDS_SPEC = {"gc": ["gc"], "expression": ["log_basemean"], ...}

-- input paths derived from __file__, the study's own baseline AUROC, and the
study's column schema. None of it was reachable from the command line, so the
module ran against exactly one dataset in exactly one repo layout. The logic in
between was general the whole time.

Here the caller passes the frame, the probabilities, the spec, and the anchor.
Nothing is read from disk and nothing is written to it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .balance import BalanceReport, balance_report
from .matching import match_negatives
from .metrics import AurocCI, bootstrap_auroc, recovery, verdict


@dataclass(frozen=True)
class ConfoundResult:
    name: str
    ci: AurocCI
    recovery: float
    verdict: str
    n_positives: int
    n_matched_negatives: int
    match_complete: bool
    match_selective: bool
    unmatched_positions: list = field(default_factory=list)
    # None only on results built by hand; `evaluate_confound` always fills it.
    balance: BalanceReport | None = None


def evaluate_confound(
    df: pd.DataFrame,
    prob_map: Mapping,
    columns: Sequence[str],
    anchor: float,
    *,
    name: str = "confound",
    id_column: str = "id",
    label_column: str = "label",
    recovery_threshold: float = 0.70,
    method: str = "nearest",
    caliper: float | None = None,
    max_smd: float | None = 0.1,
    smd_denominator: str = "pooled",
    require_complete_match: bool | None = None,
    require_selective_match: bool = True,
    bootstrap_n: int = 2000,
    seed: int = 42,
) -> ConfoundResult:
    """Match negatives to positives on `columns`, then re-score on the matched pairs.

    `max_smd` is the balance the match must reach before its AUROC means
    anything: above it the matched set is still confounded, and a classifier
    that separates it has survived nothing. None reports the imbalance in
    `.balance` instead of raising. A `caliper` is how balance is bought -- by
    dropping the positives that have no close negative -- so with one set an
    incomplete match is the design, not a failure: `require_complete_match`
    defaults to True without a caliper and False with one.

    Only matched positives are scored. 0.3.2 scored every positive against the
    fewer matched negatives, which put the unmatched positives -- the ones in the
    region the pool could not cover -- back into a comparison that claimed to be
    1:1 matched.
    """
    if require_complete_match is None:
        require_complete_match = caliper is None
    missing = [c for c in list(columns) + [id_column, label_column] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: columns not in frame: {missing}. Present: {list(df.columns)[:12]}"
        )

    pos = df[df[label_column] == 1]
    neg = df[df[label_column] == 0]
    if pos.empty or neg.empty:
        raise ValueError(
            f"{name}: need both classes; got {len(pos)} positive and {len(neg)} negative rows"
        )

    dupes = df[id_column][df[id_column].duplicated()].unique()
    if len(dupes):
        raise ValueError(
            f"{name}: {len(dupes)} duplicate id(s) in column '{id_column}' "
            f"(e.g. {list(dupes[:3])}). An id must name exactly one row: "
            f"probabilities are looked up by id and the evaluated set is rebuilt "
            f"by id, so a duplicate silently drags its twin into the evaluation "
            f"and the reported 1:1 design is not 1:1."
        )

    match = match_negatives(
        pos[list(columns)].to_numpy(),
        neg[id_column].tolist(),
        neg[list(columns)].to_numpy(),
        method=method,
        caliper=caliper,
    )
    if require_complete_match:
        match.require_complete()
    if require_selective_match:
        match.require_selective()
    if not match.matched_ids:
        raise ValueError(
            f"{name}: no positive found a negative"
            + (f" within caliper {caliper}" if caliper is not None else "")
            + "; there is nothing to evaluate"
        )

    matched_pos = pos.iloc[match.matched_positions]
    matched_neg = neg.set_index(id_column, drop=False).loc[match.matched_ids]
    balance = balance_report(
        pos[list(columns)].to_numpy(),
        neg[list(columns)].to_numpy(),
        matched_pos[list(columns)].to_numpy(),
        matched_neg[list(columns)].to_numpy(),
        columns=list(columns),
        denominator=smd_denominator,
    )
    if max_smd is not None:
        balance.require_balanced(max_smd, name=name)

    eval_ids = set(matched_pos[id_column].tolist()) | set(match.matched_ids)
    sub = df[df[id_column].isin(eval_ids)]

    unknown = [g for g in sub[id_column] if g not in prob_map]
    if unknown:
        raise ValueError(
            f"{name}: {len(unknown)} evaluated ids have no probability "
            f"(e.g. {unknown[:3]}). Scoring them as anything would invent data."
        )

    y = sub[label_column].to_numpy()
    p = np.array([prob_map[g] for g in sub[id_column]])
    ci = bootstrap_auroc(y, p, n=bootstrap_n, seed=seed)

    return ConfoundResult(
        name=name,
        ci=ci,
        recovery=recovery(ci.point, anchor),
        verdict=verdict(ci, anchor, recovery_threshold),
        n_positives=int((y == 1).sum()),
        n_matched_negatives=int((y == 0).sum()),
        match_complete=match.complete,
        match_selective=match.selective,
        unmatched_positions=list(match.unmatched_positions),
        balance=balance,
    )


def run_battery(
    df: pd.DataFrame,
    prob_map: Mapping,
    spec: Mapping[str, Sequence[str]],
    anchor: float,
    **kwargs,
) -> dict[str, ConfoundResult]:
    """Evaluate every confound in `spec`. Keys name the confound, values its columns."""
    if not spec:
        raise ValueError("spec is empty; there is nothing to control for")
    return {
        name: evaluate_confound(df, prob_map, cols, anchor, name=name, **kwargs)
        for name, cols in spec.items()
    }


def battery_passes(results: Mapping[str, ConfoundResult]) -> bool:
    """True only if every confound came back robust.

    The source computed this over a hardcoded subset of its own confound names
    (`univariate + ["joint"]`), so a confound added to the spec was scored,
    printed, and then left out of the pass/fail decision. Here every result in
    the battery counts -- a confound you bothered to declare cannot be quietly
    excluded from the verdict it exists to inform.
    """
    if not results:
        raise ValueError("no results; an empty battery cannot pass")
    from .metrics import ROBUST

    return all(r.verdict == ROBUST for r in results.values())


def format_battery(results: Mapping[str, ConfoundResult], anchor: float) -> str:
    lines = [f"confound battery (anchor AUROC={anchor:.4f})", ""]
    for name, r in results.items():
        flag = (
            ""
            if r.match_complete
            else f"  [INCOMPLETE MATCH: {len(r.unmatched_positions)} positives left out]"
        )
        if not r.match_selective:
            flag += "  [VACUOUS CONTROL: whole pool used]"
        if r.balance is not None and not r.balance.balanced():
            flag += f"  [IMBALANCED: max |SMD| = {r.balance.max_abs_smd_after:.3f}]"
        lines.append(
            f"  {name:18s} AUROC={r.ci.point:.4f} "
            f"CI[{r.ci.lo:.4f},{r.ci.hi:.4f}] "
            f"recovery={r.recovery:.2f} npos={r.n_positives} "
            f"-> {r.verdict}{flag}"
        )
    lines += ["", f"BATTERY PASS: {battery_passes(results)}"]
    return "\n".join(lines)
