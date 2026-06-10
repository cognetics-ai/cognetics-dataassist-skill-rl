"""Experience-based SQL skill distillation (proposal Section 4.1, Equations 4–5).

The teacher model M_T is applied to successful and failed trajectories to extract:
  s+ = M_T(τ+, d)  -- strategic SQL patterns from successes
  s- = M_T(τ-, d)  -- failure lessons / repair rules from failures

A failed trajectory is never injected raw; it is compressed into a typed
counterfactual that names the error class. The natural SQL error classes
(invalid_identifier, wrong_join_grain, missing_group_by, null_mishandling,
window_function_misuse, dialect_mismatch) make this compression unusually
well-posed.

In this implementation the teacher is invoked via a tool-capable LLM (configured
in settings as TEACHER_MODEL). For offline/test use, a rule-based fallback
``RuleBasedTeacher`` derives lessons directly from the gate report without an LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..observability.logging import get_logger
from ..verification.reward import RewardConfig
from .rollout import Trajectory

log = get_logger(__name__)

# ── Distilled skill record (pre-embedding, ready for catalog.repository.add_skill)
@dataclass
class DistilledSkill:
    scope: str          # "schema_specific" | "failure_repair" | "verifier_obligation"
    title: str
    principle: str
    when_to_apply: str | None = None
    positive_example: str | None = None
    negative_example: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    dialect: str | None = None
    source_id: str | None = None


# ── Error class detection from gate reports ────────────────────────────────────
_GATE_TO_CLASS = {
    "safe": "dialect_unsafe_statement",
    "parse": "dialect_mismatch",
    "bind": "invalid_identifier",
}


def _error_class_from_trajectory(traj: Trajectory) -> str:
    if traj.reward.gate_report is not None:
        ff = traj.reward.gate_report.first_failure
        if ff:
            return _GATE_TO_CLASS.get(ff, ff)
    if traj.reward.exec_result and traj.reward.exec_result.error:
        err = traj.reward.exec_result.error.lower()
        if "invalid identifier" in err or "unknown column" in err or "does not exist" in err:
            return "invalid_identifier"
        if "group by" in err:
            return "missing_group_by"
        if "null" in err:
            return "null_mishandling"
        if "syntax" in err or "parse" in err:
            return "dialect_mismatch"
        if "join" in err or "cross" in err:
            return "wrong_join_grain"
    return "result_mismatch"


# ── Rule-based teacher (no LLM required; used in tests / offline) ──────────────
class RuleBasedTeacher:
    """Derives skill records from gate reports and error messages deterministically.

    Used as a fallback when no LLM teacher is configured, or during offline
    distillation on known error classes.
    """

    _TEMPLATES: dict[str, dict[str, str]] = {
        "invalid_identifier": {
            "scope": "failure_repair",
            "title": "Repair: copy exact identifiers from schema evidence",
            "principle": (
                "On 'invalid identifier' errors, copy the exact table and column names "
                "from the linked schema context -- do not guess or infer from the question. "
                "Common cause: referencing an inner CTE source column name in a downstream "
                "CTE that should use the output alias."
            ),
            "when_to_apply": "Error: unknown column / invalid identifier / does not exist.",
        },
        "missing_group_by": {
            "scope": "failure_repair",
            "title": "Repair: add missing GROUP BY columns",
            "principle": (
                "Every non-aggregated column in the SELECT list must appear in the GROUP BY. "
                "Add the missing columns or wrap them in an aggregate function."
            ),
            "when_to_apply": "Error: not in GROUP BY / must appear in GROUP BY.",
        },
        "null_mishandling": {
            "scope": "failure_repair",
            "title": "Repair: handle NULL explicitly with IS NULL / COALESCE",
            "principle": (
                "Replace '= NULL' with 'IS NULL'. Wrap nullable expressions in "
                "COALESCE(expr, default) to avoid silent NULL propagation. "
                "Use NULLIF for zero-division guards."
            ),
            "when_to_apply": "Result mismatch when NULLable columns are involved.",
        },
        "dialect_mismatch": {
            "scope": "failure_repair",
            "title": "Repair: align syntax to target dialect",
            "principle": (
                "The SQL uses syntax from the wrong dialect. Check: QUALIFY vs subquery "
                "for window filters (Snowflake), ILIKE vs LOWER/LIKE, GENERATOR vs "
                "generate_series, :: cast vs CAST(), LATERAL FLATTEN vs unnest."
            ),
            "when_to_apply": "Parse or compilation error in the target dialect.",
        },
        "wrong_join_grain": {
            "scope": "failure_repair",
            "title": "Repair: add join predicates to prevent Cartesian products",
            "principle": (
                "A JOIN without an ON or USING clause produces a Cartesian product. "
                "Add a binding predicate that links the joined tables on a foreign-key "
                "or business key."
            ),
            "when_to_apply": "Unintended cross join / result row count far exceeds expectation.",
        },
        "result_mismatch": {
            "scope": "failure_repair",
            "title": "Repair: verify output grain against question intent",
            "principle": (
                "When the query executes but the result does not match the gold, "
                "re-check: (1) the GROUP BY grain, (2) filter conditions, "
                "(3) join direction (LEFT vs INNER), (4) date range boundaries."
            ),
            "when_to_apply": "Query executes but produces wrong rows/values.",
        },
    }

    def distill_failure(self, traj: Trajectory) -> DistilledSkill | None:
        error_class = _error_class_from_trajectory(traj)
        template = self._TEMPLATES.get(error_class)
        if template is None:
            return None
        return DistilledSkill(
            **template,
            provenance={"source": "rule_based_teacher", "error_class": error_class,
                        "task_id": traj.task_id},
            positive_example=None,
            negative_example=traj.sql,
        )

    def distill_success(self, traj: Trajectory) -> DistilledSkill | None:
        # Rule-based teacher can't extract a positive SQL strategy without understanding
        # the question; return None and let the LLM teacher handle it.
        return None


# ── LLM-based teacher ─────────────────────────────────────────────────────────
class LLMTeacher:
    """Calls a tool-capable model to extract structured skill records from trajectories.

    Uses the LiteLLM client to call the configured TEACHER_MODEL. The model must
    return a JSON object matching :class:`DistilledSkill`.
    """

    _FAILURE_PROMPT = """\
You are a SQL expert analyzing a failed Text-to-SQL attempt.

Question: {question}
Dialect: {dialect}
Generated SQL:
{sql}

Error / failure stage: {error}

Write a failure lesson as a JSON object with exactly these keys:
  scope        -- one of: failure_repair, schema_specific, verifier_obligation
  title        -- short title (under 60 chars)
  principle    -- actionable rule to avoid this failure (2-4 sentences)
  when_to_apply -- trigger condition (1 sentence)
  negative_example -- the flawed SQL above, optionally annotated

Return ONLY valid JSON. No explanation.
"""

    _SUCCESS_PROMPT = """\
You are a SQL expert analyzing a successful Text-to-SQL attempt.

Question: {question}
Dialect: {dialect}
Generated SQL:
{sql}

Write a success skill as a JSON object with exactly these keys:
  scope        -- one of: schema_specific, general_sql
  title        -- short title (under 60 chars)
  principle    -- the key decision that made this query correct (2-4 sentences)
  when_to_apply -- trigger condition (1 sentence)
  positive_example -- the SQL above, optionally annotated

Return ONLY valid JSON. No explanation.
"""

    def __init__(self, model: str = "gemini-2.5-pro", api_base: str | None = None) -> None:
        self.model = model
        self.api_base = api_base

    def _call(self, prompt: str) -> dict[str, Any] | None:
        try:
            import litellm  # lazy import

            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            log.warning("llm_teacher_failed", error=str(e))
            return None

    def distill_failure(self, traj: Trajectory) -> DistilledSkill | None:
        error = (
            (traj.reward.exec_result.error or "execution error")
            if traj.reward.exec_result
            else f"stage={traj.reward.stage}"
        )
        prompt = self._FAILURE_PROMPT.format(
            question=traj.question,
            dialect=getattr(traj.reward.exec_result, "dialect", "snowflake"),
            sql=traj.sql,
            error=error,
        )
        data = self._call(prompt)
        if data is None:
            return RuleBasedTeacher().distill_failure(traj)
        return DistilledSkill(
            **{k: data.get(k) for k in ("scope", "title", "principle", "when_to_apply",
                                          "positive_example", "negative_example")
               if data.get(k)},
            provenance={"source": "llm_teacher", "model": self.model, "task_id": traj.task_id},
        )

    def distill_success(self, traj: Trajectory) -> DistilledSkill | None:
        prompt = self._SUCCESS_PROMPT.format(
            question=traj.question,
            dialect=getattr(traj.reward.exec_result, "dialect", "snowflake"),
            sql=traj.sql,
        )
        data = self._call(prompt)
        if data is None:
            return None
        return DistilledSkill(
            **{k: data.get(k) for k in ("scope", "title", "principle", "when_to_apply",
                                          "positive_example", "negative_example")
               if data.get(k)},
            provenance={"source": "llm_teacher", "model": self.model, "task_id": traj.task_id},
        )


# ── Public distillation entry point ───────────────────────────────────────────
def distill_trajectories(
    trajectories_success: list[Trajectory],
    trajectories_failure: list[Trajectory],
    teacher: RuleBasedTeacher | LLMTeacher | None = None,
    config: RewardConfig | None = None,
) -> list[DistilledSkill]:
    """Apply the teacher to T+ and T- and return the merged list of new skills."""
    teacher = teacher or RuleBasedTeacher()
    skills: list[DistilledSkill] = []

    for traj in trajectories_success:
        sk = teacher.distill_success(traj)
        if sk is not None:
            _attach_trajectory_provenance(sk, traj)
            skills.append(sk)

    for traj in trajectories_failure:
        sk = teacher.distill_failure(traj)
        if sk is not None:
            _attach_trajectory_provenance(sk, traj)
            skills.append(sk)

    log.info("distillation_done", total=len(skills),
             successes=len(trajectories_success), failures=len(trajectories_failure))
    return skills


def _attach_trajectory_provenance(skill: DistilledSkill, traj: Trajectory) -> None:
    if traj.source_id and not skill.source_id:
        skill.source_id = str(traj.source_id)
    skill.provenance.setdefault("task_id", traj.task_id)
    if traj.source_id:
        skill.provenance.setdefault("source_id", str(traj.source_id))
