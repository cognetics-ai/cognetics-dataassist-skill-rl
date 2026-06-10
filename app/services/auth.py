from __future__ import annotations

from typing import Any

from app.services.directory import DirectoryService


class AuthService:
    """Authentication facade backed by directory lookup."""

    def __init__(self, directory: DirectoryService):
        self._directory = directory

    async def authenticate(self, soeid: str, password: str) -> dict[str, Any]:
        """Authenticate user with temporary hardcoded password policy.

        Current non-SSO policy:
            - Any non-empty SOEID is allowed if password equals "test".
        """

        if password != "test":
            raise PermissionError("Invalid credentials")
        return await self.me(soeid)

    async def me(self, employee_id: str | None) -> dict[str, Any]:
        """Resolve authenticated user profile from directory service."""

        employee_id_value = (employee_id or "").strip()
        if not employee_id_value:
            raise ValueError("employee_id_value is required")

        payload = await self._directory.get_user_directory_information(employee_id_value)
        info = payload.get("UserDirectoryInformation", {})
        work = info.get("UsersWorkInformation", {}) if isinstance(info, dict) else {}

        resolved_soeid = str(info.get("UsersID") or soeid_value).strip()
        role = await self._directory.get_user_role(resolved_soeid)
        name = str(info.get("Name") or "").strip()
        job_title = str(work.get("JobProfileTitle") or work.get("BusinessTitle") or "").strip()

        return {
            "soeid": resolved_soeid,
            "role": role,
            "display_name": name or resolved_soeid,
            "email": str(info.get("Email") or "").strip(),
            "job_title": job_title,
        }
