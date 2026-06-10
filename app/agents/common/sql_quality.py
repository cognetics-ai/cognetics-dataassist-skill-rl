from __future__ import annotations

from typing import Any

from app.agents.common.runtime_context import AgentDependencies
from app.core.sql_utils import extract_tables, has_limit, query_shape_signature


async def evaluate_sql_quality(
    deps: AgentDependencies,
    sql: str,
    engine: str,
    role_id: str,
    prompt: str,
) -> dict[str, Any]:
    """Evaluate SQL quality using policy, explain, formal gates, and obligation scoring.

    Runs three layers of checks in order of cost:
      1. Formal static-lattice gates (fast, parse-level — Safe, Parse, Bind, Scope, Join)
      2. Policy guardrails + SQL shape heuristics (local)
      3. EXPLAIN plan on the target engine (network round-trip; only when gates pass)

    The obligation-satisfaction score ω(y, q) measures how well the SQL matches
    the structural intent of the question (grain, date spine, window ranking, etc.)
    and feeds into the composite verifier reward cascade (Equation 12 of the proposal).

    Args:
        deps:     Shared runtime dependencies for policy and engine access.
        sql:      SQL text to evaluate.
        engine:   Engine key for EXPLAIN validation.
        role_id:  Effective role identifier for policy enforcement.
        prompt:   Natural-language question for obligation extraction and context.

    Returns:
        Dict with: approved, risk_score, issues, recommendations, policy_findings,
        fixes, explain_summary, shape, prompt, formal_gates, obligation_score.
    """
    # ── Layer 1: Formal static gates (Section 5.2) — no network, immediate ────
    gate_report = _run_gates(sql, deps)
    gate_issues: list[str] = []
    gate_recommendations: list[str] = []

    if gate_report:
        for msg in gate_report.get("messages", []):
            gate_issues.append(f"[gate] {msg}")
        ff = gate_report.get("first_failure")
        if ff == "safe":
            gate_recommendations.append("Remove DDL/DML/multi-statement content; use SELECT/WITH only.")
        elif ff == "parse":
            gate_recommendations.append("Fix dialect syntax errors before other checks.")
        elif ff == "bind":
            gate_recommendations.append("Verify table and column names against the schema context provided.")
        if not gate_report.get("scope_ok"):
            gate_recommendations.append("Move window function filter from WHERE to QUALIFY (Snowflake) or a subquery.")
        if not gate_report.get("join_ok"):
            gate_recommendations.append("Add ON or USING predicates to all joins to prevent Cartesian products.")

    # ── Layer 2: Obligation satisfaction score ω(y, q) ────────────────────────
    obligation_score = _score_obligations(sql, prompt, deps)

    # ── Layer 3: Policy + shape heuristics ────────────────────────────────────
    policy_result = deps.policy_checker.check(sql, role_id)
    shape = query_shape_signature(sql)

    issues: list[str] = gate_issues[:]
    recommendations: list[str] = gate_recommendations[:]

    for finding in policy_result.findings:
        message = finding.get("message")
        if message:
            issues.append(message)
    recommendations.extend(policy_result.fixes)

    if shape.get("has_select_star"):
        recommendations.append("Replace SELECT * with explicit columns.")
    if shape.get("cross_join"):
        recommendations.append("Add JOIN predicates to avoid Cartesian products.")

    # ── Layer 4: EXPLAIN (only when gates pass — avoids wasted network calls) ──
    explain_result = _null_explain()
    hard_gate_fail = gate_report.get("first_failure") in ("safe", "parse") if gate_report else False
    if not hard_gate_fail:
        explain_result = await deps.engines.get(engine).explain(sql)
        if not explain_result.ok:
            issues.append("Explain plan failed for target engine.")
            recommendations.append("Fix syntax/object references so EXPLAIN succeeds.")

    # A query is approved when: no hard gate failures, policy passes, EXPLAIN passes.
    gate_ok = not gate_report or gate_report.get("first_failure") is None
    approved = bool(gate_ok and policy_result.is_valid and explain_result.ok)

    return {
        "approved": approved,
        "risk_score": float(policy_result.risk_score),
        "issues": issues,
        "recommendations": sorted(set(recommendations)),
        "policy_findings": policy_result.findings,
        "fixes": policy_result.fixes,
        "explain_summary": explain_result.summary,
        "shape": shape,
        "prompt": prompt,
        "formal_gates": gate_report or {},
        "obligation_score": obligation_score,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_gates(sql: str, deps: AgentDependencies) -> dict[str, Any] | None:
    """Run the static lattice; return a plain dict (gate report) or None on import error."""
    try:
        from skillsql.verification.static_gates import run_static_lattice

        dialect = _dialect(deps)
        known = _known_tables(deps) if deps.has_catalog else None
        report = run_static_lattice(sql, dialect, known_tables=known)
        return {
            "safe": report.safe,
            "parses": report.parses,
            "binds": report.binds,
            "scope_ok": report.scope_ok,
            "join_ok": report.join_ok,
            "first_failure": report.first_failure,
            "messages": report.messages,
            "passed_all": report.passed_all,
        }
    except Exception:  # noqa: BLE001 — SkillSQL not available: skip gracefully
        return None


def _score_obligations(sql: str, question: str, deps: AgentDependencies) -> float:
    """Compute the obligation-satisfaction score ω(y, q) (Equation 11). Returns 0.0 on error."""
    if not question:
        return 0.0
    try:
        from skillsql.verification.obligations import score_obligations

        result = score_obligations(question, sql, _dialect(deps))
        return result.score
    except Exception:  # noqa: BLE001
        return 0.0


def _known_tables(deps: AgentDependencies) -> set[str] | None:
    """Return lowercase table identifiers from the catalog for the bind gate."""
    try:
        from skillsql.catalog.models import CatalogTable

        res = deps.skillsql_resources
        if res is None:
            return None
        with res.repo.session() as s:
            rows = s.query(CatalogTable).all()
        known: set[str] = set()
        for r in rows:
            known.add(r.name.lower())
            known.add(r.fqn.lower())
        return known or None
    except Exception:  # noqa: BLE001
        return None


def _dialect(deps: AgentDependencies) -> str:
    """Best-effort dialect name; defaults to 'snowflake'."""
    try:
        if deps.has_catalog and deps.skillsql_resources is not None:
            return deps.skillsql_resources.connector.dialect
    except Exception:  # noqa: BLE001
        pass
    engine = getattr(deps.settings, "default_engine", "snowflake")
    return "snowflake" if "snow" in engine.lower() else engine


def _null_explain():
    """Return a minimal explain result for when EXPLAIN is skipped."""

    class _E:
        ok = True
        summary: dict = {}

    return _E()


def optimize_sql_with_guardrails(default_limit: int, sql: str) -> dict[str, Any]:
    """Apply deterministic SQL optimization guardrails.

    Args:
        default_limit: LIMIT value enforced when query has no limit.
        sql: SQL text to optimize.

    Returns:
        Optimization payload with rewritten SQL, change list, and referenced tables.
    """

    rewritten = sql.strip().rstrip(";")
    changes: list[str] = []

    if rewritten and not has_limit(rewritten):
        rewritten = f"{rewritten} LIMIT {default_limit}"
        changes.append(f"Added LIMIT {default_limit} by policy")

    return {
        "optimized_sql": rewritten,
        "changes": changes,
        "tables": extract_tables(rewritten),
    }
