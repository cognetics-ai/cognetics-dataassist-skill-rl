from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().split())


def parse_sql(sql: str) -> exp.Expression:
    return sqlglot.parse_one(sql, read="trino")


def extract_tables(sql: str) -> list[str]:
    try:
        tree = parse_sql(sql)
    except Exception:
        return []
    tables: list[str] = []
    for node in tree.find_all(exp.Table):
        tables.append(node.sql(dialect="trino"))
    return sorted(set(tables))


def has_limit(sql: str) -> bool:
    try:
        tree = parse_sql(sql)
    except Exception:
        return False
    return tree.find(exp.Limit) is not None


def statement_kind(sql: str) -> str:
    try:
        tree = parse_sql(sql)
    except Exception:
        token = sql.strip().split(maxsplit=1)[0].upper() if sql.strip() else "UNKNOWN"
        return token
    return tree.key.upper()


def has_select_star(sql: str) -> bool:
    try:
        tree = parse_sql(sql)
    except Exception:
        return bool(re.search(r"select\s+\*", sql, flags=re.IGNORECASE))

    for select in tree.find_all(exp.Select):
        if any(isinstance(expr, exp.Star) for expr in select.expressions):
            return True
    return False


def detects_cross_join(sql: str) -> bool:
    try:
        tree = parse_sql(sql)
    except Exception:
        return bool(re.search(r"cross\s+join", sql, flags=re.IGNORECASE))

    for join in tree.find_all(exp.Join):
        kind = (join.args.get("kind") or "").upper()
        if kind == "CROSS":
            return True
        has_predicate = join.args.get("on") is not None or join.args.get("using") is not None
        if not has_predicate and kind in {"", "INNER", "JOIN"}:
            return True
    return False


def has_blocked_keyword(sql: str, blocked_keywords: list[str]) -> str | None:
    sql_norm = normalize_sql(sql).upper()
    for keyword in blocked_keywords:
        token = keyword.strip().upper()
        if not token:
            continue
        if re.search(rf"\b{re.escape(token)}\b", sql_norm):
            return token
    return None


def query_shape_signature(sql: str) -> dict[str, Any]:
    return {
        "kind": statement_kind(sql),
        "tables": extract_tables(sql),
        "has_limit": has_limit(sql),
        "has_select_star": has_select_star(sql),
        "cross_join": detects_cross_join(sql),
    }
