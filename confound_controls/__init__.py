"""Controls for "did my classifier learn the signal, or a confound?"

Extracted from the aim3 control battery shared byte-identically between
two internal research projects.
"""

from .ablation import (
    INCONCLUSIVE as ABLATION_INCONCLUSIVE,
)
from .ablation import (
    STRUCTURE_DEPENDENT,
    STRUCTURE_INDEPENDENT,
    AblationResult,
    ablation_control,
    assert_ablation_changed_input,
)
from .balance import BalanceReport, balance_report, standardized_mean_differences
from .battery import (
    ConfoundResult,
    battery_passes,
    evaluate_confound,
    format_battery,
    run_battery,
)
from .incremental import (
    ADDS,
    HARMS,
    NO_EFFECT,
    UNDERPOWERED,
    DeltaResult,
    incremental_validity,
    paired_delta_auroc,
)
from .matching import MatchResult, match_negatives, propensity_logit
from .metrics import (
    DRIVEN,
    INCONCLUSIVE,
    INVERTED,
    PARTIAL,
    ROBUST,
    AurocCI,
    bootstrap_auroc,
    recovery,
    verdict,
)
from .sequence import (
    CONFIRMED,
    NOT_CONFIRMED,
    ControlSpan,
    GroupedDelta,
    ShuffleResult,
    confirm_knockout,
    dinucleotide_counts,
    dinucleotide_shuffle,
    grouped_delta_ci,
    knockout_span,
    sample_control_span,
)

__version__ = "0.4.0"

__all__ = [
    "ABLATION_INCONCLUSIVE",
    "ADDS",
    "CONFIRMED",
    "DRIVEN",
    "HARMS",
    "INCONCLUSIVE",
    "INVERTED",
    "NOT_CONFIRMED",
    "NO_EFFECT",
    "PARTIAL",
    "ROBUST",
    "STRUCTURE_DEPENDENT",
    "STRUCTURE_INDEPENDENT",
    "UNDERPOWERED",
    "AblationResult",
    "AurocCI",
    "BalanceReport",
    "ConfoundResult",
    "ControlSpan",
    "DeltaResult",
    "GroupedDelta",
    "MatchResult",
    "ShuffleResult",
    "ablation_control",
    "assert_ablation_changed_input",
    "balance_report",
    "battery_passes",
    "bootstrap_auroc",
    "confirm_knockout",
    "dinucleotide_counts",
    "dinucleotide_shuffle",
    "evaluate_confound",
    "format_battery",
    "grouped_delta_ci",
    "incremental_validity",
    "knockout_span",
    "match_negatives",
    "paired_delta_auroc",
    "propensity_logit",
    "recovery",
    "run_battery",
    "sample_control_span",
    "standardized_mean_differences",
    "verdict",
]
