"""Semantic obligations (proposal Section 5.3).

We map natural-language cues in the question to required SQL structure, then score
the fraction of obligations a candidate satisfies. Obligations supply *dense
partial credit* that shapes learning before exact matches become common. Each
check is a decidable structural test on the parsed SQL.

This is intentionally lightweight and rule-based: it is a reward-shaping signal,
never the arbiter of correctness (exact execution match dominates the reward; see
:mod:`skillsql.verification.reward`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import sqlglot
import sqlglot.expressions as exp

Obligation = str


@dataclass
class ObligationResult:
    required: list[Obligation]
    satisfied: list[Obligation]

    @property
    def score(self) -> float:
        if not self.required:
            return 0.0
        return len(self.satisfied) / len(self.required)


# ---- NL cue patterns -> obligation labels -------------------------------------
_CUES: list[tuple[str, Obligation]] = [
    (r"\bper (month|day|week|quarter|year)\b.*\b(no|without|zero)\b", "date_spine_left_join"),
    (r"\b(each|every|per)\b.*\b(month|day|week|quarter|year)\b", "group_by_period"),
    (r"\b(latest|most recent|top|first|last|earliest|highest|lowest|max(imum)?|min(imum)?)\b.*\b(per|for each|by)\b", "window_rank_per_group"),
    (r"\b(average|avg|mean|total|sum|count|median)\b", "aggregation_grain"),
    (r"\b(net|signed|after (refunds|returns|discount))\b", "signed_amount"),
    (r"\b(running|cumulative|rolling|moving)\b", "window_cumulative"),
    (r"\b(rank|ranking|nth|top\s+\d+)\b", "ranking"),
    (r"\b(percentage|percent|ratio|share|proportion)\b", "ratio_expression"),
    (r"\bexcluding\b|\bother than\b|\bnot\b.*\bin\b", "exclusion_filter"),
    (r"\bdistinct\b|\bunique\b", "distinct"),
]


def extract_obligations(question: str) -> list[Obligation]:
    q = question.lower()
    found: list[Obligation] = []
    for pattern, label in _CUES:
        if re.search(pattern, q) and label not in found:
            found.append(label)
    return found


def _parse(sql: str, dialect: str) -> exp.Expression | None:
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except Exception:  # noqa: BLE001
        return None
    return statements[0] if statements else None


# ---- structural checks (parse tree -> bool) -----------------------------------
def _has(node: exp.Expression, t) -> bool:
    return node.find(t) is not None


def _check_date_spine_left_join(tree: exp.Expression) -> bool:
    has_left = any(
        (j.args.get("side") or "").upper() == "LEFT" for j in tree.find_all(exp.Join)
    )
    has_series = bool(
        re.search(r"generate_date|generate_series|date_spine|seq4|generator", tree.sql().lower())
    )
    return has_left and (has_series or _has(tree, exp.With))


def _check_group_by_period(tree: exp.Expression) -> bool:
    has_group = _has(tree, exp.Group)
    has_trunc = bool(re.search(r"date_trunc|trunc\(|to_char|extract", tree.sql().lower()))
    return has_group and has_trunc


def _check_window_rank(tree: exp.Expression) -> bool:
    for w in tree.find_all(exp.Window):
        fn = (w.this.sql().lower() if w.this else "")
        if any(k in fn for k in ("row_number", "rank", "dense_rank", "first_value", "last_value")):
            if w.args.get("partition_by"):
                return True
    return False


def _check_aggregation_grain(tree: exp.Expression) -> bool:
    agg = any(
        isinstance(f, exp.AggFunc) for f in tree.find_all(exp.AggFunc)
    ) or bool(re.search(r"\b(sum|avg|count|min|max|median)\s*\(", tree.sql().lower()))
    if not agg:
        return False
    # If non-aggregated columns are selected, require a GROUP BY.
    select = tree.find(exp.Select)
    if select is None:
        return agg
    bare_cols = [
        e for e in select.expressions
        if e.find(exp.Column) is not None and e.find(exp.AggFunc) is None
    ]
    if bare_cols and not _has(tree, exp.Group):
        return False
    return True


def _check_signed_amount(tree: exp.Expression) -> bool:
    s = tree.sql().lower()
    return any(tok in s for tok in ("case when", "-", "sign(", "abs(", "*-1", "* -1"))


def _check_window_cumulative(tree: exp.Expression) -> bool:
    for w in tree.find_all(exp.Window):
        if w.args.get("order") or "rows between" in w.sql().lower() or "range between" in w.sql().lower():
            fn = (w.this.sql().lower() if w.this else "")
            if "sum" in fn or "avg" in fn or "count" in fn:
                return True
    return False


def _check_ranking(tree: exp.Expression) -> bool:
    return _check_window_rank(tree) or bool(re.search(r"\bqualify\b|\blimit\b", tree.sql().lower()))


def _check_ratio(tree: exp.Expression) -> bool:
    return isinstance(tree.find(exp.Div), exp.Div) or "/" in tree.sql()


def _check_exclusion(tree: exp.Expression) -> bool:
    s = tree.sql().lower()
    return "not in" in s or "not exists" in s or "<>" in s or "!=" in s or " except " in s


def _check_distinct(tree: exp.Expression) -> bool:
    return _has(tree, exp.Distinct) or "distinct" in tree.sql().lower()


_CHECKS: dict[Obligation, Callable[[exp.Expression], bool]] = {
    "date_spine_left_join": _check_date_spine_left_join,
    "group_by_period": _check_group_by_period,
    "window_rank_per_group": _check_window_rank,
    "aggregation_grain": _check_aggregation_grain,
    "signed_amount": _check_signed_amount,
    "window_cumulative": _check_window_cumulative,
    "ranking": _check_ranking,
    "ratio_expression": _check_ratio,
    "exclusion_filter": _check_exclusion,
    "distinct": _check_distinct,
}


def score_obligations(question: str, sql: str, dialect: str) -> ObligationResult:
    """Return required vs. satisfied obligations and the resulting score in [0, 1]."""
    required = extract_obligations(question)
    tree = _parse(sql, dialect)
    if tree is None or not required:
        return ObligationResult(required=required, satisfied=[])
    satisfied = [ob for ob in required if _CHECKS.get(ob, lambda _t: False)(tree)]
    return ObligationResult(required=required, satisfied=satisfied)
