"""Controls for "did my classifier learn the signal, or a confound?"

Extracted from the aim3 control battery shared byte-identically between
arf_promoter_analysis and phelipanche-fm.
"""

from .matching import MatchResult, match_negatives
from .metrics import (
    DRIVEN,
    INCONCLUSIVE,
    PARTIAL,
    ROBUST,
    AurocCI,
    bootstrap_auroc,
    recovery,
    verdict,
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
from .ablation import (
    STRUCTURE_DEPENDENT,
    STRUCTURE_INDEPENDENT,
    INCONCLUSIVE as ABLATION_INCONCLUSIVE,
    AblationResult,
    ablation_control,
    assert_ablation_changed_input,
)
from .battery import (
    ConfoundResult,
    battery_passes,
    evaluate_confound,
    format_battery,
    run_battery,
)

__version__ = "0.2.0"

__all__ = [
    "MatchResult", "match_negatives",
    "AurocCI", "bootstrap_auroc", "recovery", "verdict",
    "ROBUST", "DRIVEN", "PARTIAL", "INCONCLUSIVE",
    "ConfoundResult", "evaluate_confound", "run_battery",
    "battery_passes", "format_battery",
    "DeltaResult", "paired_delta_auroc", "incremental_validity",
    "ADDS", "HARMS", "NO_EFFECT", "UNDERPOWERED",
    "AblationResult", "ablation_control", "assert_ablation_changed_input",
    "STRUCTURE_DEPENDENT", "STRUCTURE_INDEPENDENT", "ABLATION_INCONCLUSIVE",
]
