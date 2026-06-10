from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.sql_utils import (
    detects_cross_join,
    has_blocked_keyword,
    has_limit,
    has_select_star,
    statement_kind,
)


@dataclass
class PolicyCheckResult:
    is_valid: bool
    findings: list[dict]
    risk_score: float
    fixes: list[str]


class PolicyChecker:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._blocked_keywords = [item.strip() for item in settings.blocked_keywords.split(",") if item.strip()]

    def check(self, sql: str, role_id: str) -> PolicyCheckResult:
        findings: list[dict] = []
        fixes: list[str] = []
        risk = 0.0

        kind = statement_kind(sql)
        if kind not in {"SELECT", "UNION", "WITH"}:
            findings.append(
                {
                    "severity": "error",
                    "code": "READ_ONLY_ENFORCED",
                    "message": f"Only read-only statements are allowed. Found statement type: {kind}",
                }
            )
            risk += 0.7

        blocked = has_blocked_keyword(sql, self._blocked_keywords)
        if blocked:
            findings.append(
                {
                    "severity": "error",
                    "code": "BLOCKED_KEYWORD",
                    "message": f"Keyword '{blocked}' is blocked by policy.",
                }
            )
            risk += 0.8

        if not has_limit(sql):
            findings.append(
                {
                    "severity": "warning",
                    "code": "MISSING_LIMIT",
                    "message": f"Query does not contain LIMIT. Policy default is LIMIT {self._settings.default_limit}.",
                }
            )
            fixes.append(f"Add LIMIT {self._settings.default_limit}.")
            risk += 0.2

        if has_select_star(sql):
            findings.append(
                {
                    "severity": "warning",
                    "code": "SELECT_STAR",
                    "message": "Avoid SELECT * for large enterprise tables.",
                }
            )
            fixes.append("Select explicit columns.")
            risk += 0.2

        if detects_cross_join(sql):
            findings.append(
                {
                    "severity": "error",
                    "code": "POTENTIAL_CARTESIAN",
                    "message": "Potential cartesian join detected (CROSS join or join without predicate).",
                }
            )
            fixes.append("Add explicit JOIN predicates.")
            risk += 0.6

        is_valid = all(item["severity"] != "error" for item in findings)
        risk_score = min(1.0, round(risk, 2))
        return PolicyCheckResult(is_valid=is_valid, findings=findings, risk_score=risk_score, fixes=fixes)
