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
from .battery import (
    ConfoundResult,
    battery_passes,
    evaluate_confound,
    format_battery,
    run_battery,
)

__version__ = "0.1.0"

__all__ = [
    "MatchResult", "match_negatives",
    "AurocCI", "bootstrap_auroc", "recovery", "verdict",
    "ROBUST", "DRIVEN", "PARTIAL", "INCONCLUSIVE",
    "ConfoundResult", "evaluate_confound", "run_battery",
    "battery_passes", "format_battery",
]
