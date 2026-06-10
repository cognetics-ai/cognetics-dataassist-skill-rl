"""SQL verification: static lattice, obligations, instance equivalence, reward."""

from .equivalence import result_equivalent, set_external_checker
from .obligations import ObligationResult, extract_obligations, score_obligations
from .reward import RewardBreakdown, RewardConfig, compute_reward
from .static_gates import GateReport, run_static_lattice

__all__ = [
    "run_static_lattice", "GateReport",
    "extract_obligations", "score_obligations", "ObligationResult",
    "result_equivalent", "set_external_checker",
    "compute_reward", "RewardConfig", "RewardBreakdown",
]
