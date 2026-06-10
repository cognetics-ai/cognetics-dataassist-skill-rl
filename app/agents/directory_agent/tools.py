from __future__ import annotations

import json
import logging
from typing import Any

from app.agents.common.runtime_context import AgentDependencies

_logger = logging.getLogger(__name__)

def build_tools(deps: AgentDependencies) -> list:
    """Build tools for directory retrieval and state initialization.

    Args:
        deps: Shared runtime dependencies, including the directory service.

    Returns:
        List of callable tools that fetch user directory profile and promote it to shared state.
    """
    _logger.debug("Building tools for directory retrieval and state initialization.")

    async def get_user_directory_info(employee_id: str, tool_context: Any | None = None) -> dict[str, Any]:
        """Fetch and summarize directory profile for a user with provided employee id.

        Args:
            employee_id: Employee id for the authenticated user. Alphanumeric string.

        Returns:
            Schema-ready payload containing `UserDirectoryInformation` and `directory_summary`.
        """

        payload = await deps.directory.get_user_directory_information(employee_id)
        role_id = await deps.directory.get_user_role(employee_id)

        info = payload.get("UserDirectoryInformation", {})
        work = info.get("UsersWorkInformation", {}) if isinstance(info, dict) else {}
        last_two = work.get("ManagedSegmentLastTwoLevels", []) if isinstance(work, dict) else []
        summary = _directory_summary(payload)

        if tool_context:
            tool_context.state["user_directory_information_json"] = json.dumps(payload)
            tool_context.state["directory_summary_json"] = json.dumps(summary)
            tool_context.state["role_id"] = role_id
            tool_context.state["user_soeid"] = str(info.get("UsersID") or employee_id)
            tool_context.state["user_email"] = str(info.get("Email") or "")
            tool_context.state["managed_segment_hierarchy"] = str(work.get("ManagedSegmentHierarchy") or "")
            tool_context.state["managed_segment_last_two_levels_json"] = json.dumps(last_two if isinstance(last_two, list) else [])
        return {
            "UserDirectoryInformation": info if isinstance(info, dict) else {},
            "directory_summary": summary,
        }

    return [get_user_directory_info]


def _directory_summary(payload: dict[str, Any]) -> dict[str, Any]:
    info = payload.get("UserDirectoryInformation", {})
    if not isinstance(info, dict):
        info = {}
    work = info.get("UsersWorkInformation", {})
    if not isinstance(work, dict):
        work = {}
    return {
        "UsersID": str(info.get("UsersID", "")),
        "Email": str(info.get("Email", "")),
        "Name": str(info.get("Name", "")),
        "JobProfileTitle": str(work.get("JobProfileTitle", "")),
        "BusinessTitle": str(work.get("BusinessTitle", "")),
        "ManagedSegmentPath": str(work.get("ManagedSegmentPath", "")),
        "ManagedGeographPath": str(work.get("ManagedGeographPath", "")),
        "ManagedSegmentLastTwoLevels": work.get("ManagedSegmentLastTwoLevels", []),
    }
