"""Instance (execution) equivalence -- the decidable reward oracle (proposal 5.1).

This is the Spider-2.0 EX notion: a predicted query is correct if, executed on the
concrete database, its result is a multiset of rows matching the reference, with
*extra predicted columns tolerated*. General (input-independent) equivalence is
undecidable; we only ever compare results on a concrete instance, which is
decidable by execution.

For leaderboard-grade scoring, plug the official Spider-2.0 checker in via
:func:`set_external_checker`; this module's matcher is a faithful, dependency-free
default suitable for training and self-evaluation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from ..connectors.base import ExecResult

# Optional external (official) checker hook.
_EXTERNAL_CHECKER: Callable[[ExecResult, ExecResult], bool] | None = None


def set_external_checker(fn: Callable[[ExecResult, ExecResult], bool] | None) -> None:
    """Register an official evaluator (e.g. Spider-2.0) to override the default."""
    global _EXTERNAL_CHECKER
    _EXTERNAL_CHECKER = fn


def _canon_cell(v: Any, float_tol_decimals: int) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return float(v) if False else int(v)
    if isinstance(v, float):
        return round(v, float_tol_decimals)
    # Decimals, dates, etc. -> normalized string
    s = str(v).strip()
    try:
        return round(float(s), float_tol_decimals)
    except (TypeError, ValueError):
        return s.lower()


def _column_multiset(rows: list[list[Any]], idx: int, decimals: int) -> Counter:
    return Counter(_canon_cell(r[idx], decimals) for r in rows if idx < len(r))


def _row_multiset(rows: list[list[Any]], idxs: list[int], decimals: int) -> Counter:
    return Counter(
        tuple(_canon_cell(r[i], decimals) for i in idxs) for r in rows if len(r) > max(idxs, default=-1)
    )


def result_equivalent(
    predicted: ExecResult,
    gold: ExecResult,
    *,
    order_insensitive: bool = True,
    float_decimals: int = 6,
    allow_extra_columns: bool = True,
) -> bool:
    """Return True iff ``predicted`` matches ``gold`` under EX semantics."""
    if _EXTERNAL_CHECKER is not None:
        return _EXTERNAL_CHECKER(predicted, gold)
    if not predicted.ok or not gold.ok:
        return False
    if not gold.rows and not predicted.rows:
        return True
    if len(gold.rows) != len(predicted.rows):
        # Row count must match (extra *columns* tolerated, not extra *rows*).
        return False

    n_gold_cols = len(gold.columns) or (len(gold.rows[0]) if gold.rows else 0)
    n_pred_cols = len(predicted.columns) or (len(predicted.rows[0]) if predicted.rows else 0)

    if not allow_extra_columns and n_gold_cols != n_pred_cols:
        return False

    # Greedily map each gold column to an unused predicted column with an identical
    # value-multiset, then verify the row-level projection matches as a multiset.
    pred_multisets = [_column_multiset(predicted.rows, j, float_decimals) for j in range(n_pred_cols)]
    used: set[int] = set()
    mapping: list[int] = []
    for gi in range(n_gold_cols):
        gms = _column_multiset(gold.rows, gi, float_decimals)
        match = next(
            (j for j in range(n_pred_cols) if j not in used and pred_multisets[j] == gms),
            None,
        )
        if match is None:
            return False
        used.add(match)
        mapping.append(match)

    if not order_insensitive:
        proj_pred = [[r[j] for j in mapping] for r in predicted.rows]
        proj_gold = [[r[i] for i in range(n_gold_cols)] for r in gold.rows]
        return [
            [_canon_cell(c, float_decimals) for c in row] for row in proj_pred
        ] == [[_canon_cell(c, float_decimals) for c in row] for row in proj_gold]

    pred_rows = _row_multiset(predicted.rows, mapping, float_decimals)
    gold_rows = _row_multiset(gold.rows, list(range(n_gold_cols)), float_decimals)
    return pred_rows == gold_rows
