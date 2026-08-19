"""Score an ablated input with the same model and see what survives.

Extracted from an internal grammar-shuffle script
(byte-identical in a second internal project). There the ablation was a
dinucleotide-preserving shuffle of promoter sequences, applied zero-shot to a
frozen probe: if AUROC collapses toward chance, the model was reading order
rather than composition.

Nothing about that argument is specific to sequences. The general shape is:
take a model's scores on real input, take its scores on input with the
structure of interest destroyed but the nuisance properties preserved, and ask
whether the difference clears zero.

The source got one thing right that is worth keeping and generalising -- it
refused to run when the ablation had not changed anything:

    assert not np.allclose(shuffled, real), \\
        "shuffled embeddings identical to real - shuffle/embeds broken"

Without that check, a broken ablation produces scores identical to the real
ones, the delta is exactly zero, the interval hugs zero, and the module reports
"the model survives ablation" -- the strongest possible result, obtained by
doing nothing. That failure is silent, and it is the same shape as the vacuous
matched control in `matching.py`: an inert control returns the original answer,
which reads as the hypothesis surviving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

# What survived the ablation.
STRUCTURE_DEPENDENT = "structure-dependent"  # ablation collapsed it to chance
STRUCTURE_INDEPENDENT = "structure-independent"  # ablated scores still separate
PARTIAL = "partial"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class AblationResult:
    auroc_real: float
    auroc_ablated: float
    delta: float
    delta_lo: float
    delta_hi: float
    ablated_lo: float
    ablated_hi: float
    verdict: str
    n_resamples_used: int

    @property
    def ablated_includes_chance(self) -> bool:
        return self.ablated_lo <= 0.5 <= self.ablated_hi


def assert_ablation_changed_input(real, ablated, *, name: str = "ablation") -> None:
    """Refuse an ablation that did not alter its input.

    Call this on the ABLATED REPRESENTATION (sequences, embeddings, features) --
    not on the scores. Identical scores can also arise from a model that
    genuinely ignores the ablated structure, which is a finding; identical
    inputs are a broken pipeline, and the two must not be confused.
    """
    real = np.asarray(real)
    ablated = np.asarray(ablated)
    if real.shape != ablated.shape:
        raise ValueError(
            f"{name}: shapes differ, real={real.shape} ablated={ablated.shape}; "
            f"these are meant to be the same inputs with structure destroyed"
        )
    if real.dtype.kind in "fc" or ablated.dtype.kind in "fc":
        identical = np.allclose(real, ablated)
    else:
        identical = np.array_equal(real, ablated)
    if identical:
        raise ValueError(
            f"{name}: the ablated input is identical to the real input, so the "
            f"ablation did nothing. Every downstream comparison would report "
            f"the model as surviving it -- the strongest possible result, "
            f"obtained by doing nothing."
        )


def ablation_control(
    y,
    p_real,
    p_ablated,
    *,
    n: int = 2000,
    seed: int = 42,
    max_ci_width: float | None = None,
) -> AblationResult:
    """Compare real vs ablated scores with a paired bootstrap.

    Both the delta interval and the ablated AUROC's own interval are reported,
    because they answer different questions: the delta says the ablation
    changed something, the ablated interval says whether anything is left.
    """
    y = np.asarray(y)
    p_real = np.asarray(p_real)
    p_ablated = np.asarray(p_ablated)
    if not (len(y) == len(p_real) == len(p_ablated)):
        raise ValueError(
            f"lengths differ: y={len(y)}, real={len(p_real)}, ablated={len(p_ablated)}"
        )
    if len(np.unique(y)) < 2:
        raise ValueError("AUROC needs both classes present in y")

    real_auc = roc_auc_score(y, p_real)
    abl_auc = roc_auc_score(y, p_ablated)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    deltas, ablated_aucs = [], []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        a = roc_auc_score(y[b], p_ablated[b])
        deltas.append(roc_auc_score(y[b], p_real[b]) - a)
        ablated_aucs.append(a)

    if len(deltas) < n // 2:
        raise ValueError(
            f"only {len(deltas)} of {n} resamples contained both classes; the "
            f"intervals would rest on too few replicates to mean anything"
        )

    d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
    a_lo, a_hi = np.percentile(ablated_aucs, [2.5, 97.5])
    delta_excludes_zero = d_lo > 0 or d_hi < 0
    ablated_includes_chance = a_lo <= 0.5 <= a_hi

    if max_ci_width is not None and (d_hi - d_lo) > max_ci_width:
        verdict = INCONCLUSIVE
    elif ablated_includes_chance and delta_excludes_zero:
        verdict = STRUCTURE_DEPENDENT
    elif a_lo > 0.5 and not delta_excludes_zero:
        # Ablated scores still separate, and the drop is not distinguishable
        # from zero: the structure was never what the model was using.
        verdict = STRUCTURE_INDEPENDENT
    else:
        verdict = PARTIAL

    return AblationResult(
        auroc_real=float(real_auc),
        auroc_ablated=float(abl_auc),
        delta=float(real_auc - abl_auc),
        delta_lo=float(d_lo),
        delta_hi=float(d_hi),
        ablated_lo=float(a_lo),
        ablated_hi=float(a_hi),
        verdict=verdict,
        n_resamples_used=len(deltas),
    )
