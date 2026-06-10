from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions


class RefinementLoopExitAgent(BaseAgent):
    """Deterministic loop controller for the critic/refiner cycle."""

    max_refinement_passes: int = 1

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state_delta, should_exit = self.build_state_delta(
            dict(ctx.session.state or {}),
            max_refinement_passes=max(1, self.max_refinement_passes),
        )
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta=state_delta,
                escalate=should_exit,
            ),
        )

    @classmethod
    def build_state_delta(
        cls,
        state: dict[str, Any],
        *,
        max_refinement_passes: int = 1,
    ) -> tuple[dict[str, Any], bool]:
        critic_package = cls._state_json(state, "critic_package_json")
        refinement_package = cls._state_json(state, "refinement_package_json")
        draft_package = cls._state_json(state, "draft_package_json")

        current_sql = str(
            state.get("submitted_sql")
            or state.get("refined_sql")
            or state.get("generated_sql")
            or draft_package.get("draft_sql")
            or ""
        ).strip()
        refined_sql = str(refinement_package.get("refined_sql") or "").strip()
        next_sql = refined_sql or current_sql

        previous_pass_count = cls._int_value(state.get("refinement_pass_count"))
        pass_count = previous_pass_count + 1

        critic_approved = bool(critic_package.get("approved"))
        model_requested_exit = bool(refinement_package.get("exit_loop"))
        pass_limit_reached = pass_count >= max(1, max_refinement_passes)
        should_exit = critic_approved or model_requested_exit or pass_limit_reached

        if critic_approved:
            exit_reason = "critic_approved"
        elif model_requested_exit:
            exit_reason = str(refinement_package.get("exit_reason") or "refiner_requested_exit")
        elif pass_limit_reached:
            exit_reason = "max_refinement_passes_reached"
        else:
            exit_reason = "continue_refinement"

        exit_payload = {
            "exit_loop": should_exit,
            "exit_reason": exit_reason,
            "critic_approved": critic_approved,
            "refinement_pass_count": pass_count,
            "max_refinement_passes": max(1, max_refinement_passes),
        }

        state_delta: dict[str, Any] = {
            "refinement_pass_count": pass_count,
            "refinement_loop_exit_json": json.dumps(exit_payload, default=str),
        }
        if next_sql:
            state_delta["submitted_sql"] = next_sql
            state_delta["refined_sql"] = next_sql

        return state_delta, should_exit

    @staticmethod
    def _state_json(state: dict[str, Any], key: str) -> dict[str, Any]:
        raw = state.get(key)
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
        return {}

    @staticmethod
    def _int_value(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0
