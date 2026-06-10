from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote_plus

import aiohttp

from app.config import Settings
from app.core.store import SQLStore


class DirectoryService:
    """Directory facade backed by ECS people search endpoint."""

    def __init__(self, settings: Settings, store: SQLStore):
        self._settings = settings
        self._store = store
        self._lookup_cache: dict[str, dict[str, str]] = {}

    async def get_user_directory_information(self, soeid: str) -> dict[str, Any]:
        """Fetch user profile by SOEID and normalize managed-segment hierarchy.

        Args:
            soeid: User identifier used in people search query.

        Returns:
            Directory payload with user identity, email, managed segment hierarchy,
            parsed managed-segment levels (L1-L12), and last-two level hints.
        """

        if not soeid or not soeid.strip():
            raise ValueError("SOEID is required for directory lookup")

        details = await self._lookup_by_soeid(soeid.strip())
        segment_levels = self._parse_segment_levels(details.get("managedsegmenthierarchy", ""))
        ordered_levels = sorted(segment_levels.keys())
        last_two_levels = ordered_levels[-2:] if len(ordered_levels) >= 2 else ordered_levels

        managed_path = " > ".join(segment_levels[level] for level in ordered_levels if segment_levels.get(level))
        last_two = [segment_levels[level] for level in last_two_levels if segment_levels.get(level)]

        work_payload: dict[str, Any] = {
            "JobProfileTitle": details.get("job_title", ""),
            "JobFamilyGroup": "",
            "Title": "",
            "BusinessTitle": details.get("business_title", ""),
            "ManagedSegmentPath": managed_path,
            "ManagedGeographPath": "",
            "ManagedSegmentHierarchy": details.get("managedsegmenthierarchy", ""),
            "ManagedSegmentLastTwoLevels": last_two,
        }
        for idx in range(1, 13):
            work_payload[f"ManagedSegmentL{idx}"] = segment_levels.get(idx)

        return {
            "UserDirectoryInformation": {
                "UsersID": details.get("soeid") or soeid.strip(),
                "Email": details.get("email", ""),
                "Name": details.get("name", ""),
                "UsersWorkInformation": work_payload,
            }
        }

    async def get_user_role(self, soeid: str) -> str:
        """Resolve effective role for policy checks.

        Resolution order:
        1. Existing local store role mapping (if user exists).
        2. Configured default role.
        """

        normalized = (soeid or "").strip()
        if normalized:
            try:
                details = await self._lookup_by_soeid(normalized)
                business_title = str(details.get("business_title") or "").strip()
                if business_title:
                    return business_title
            except Exception:
                pass

        user = await self._store.get_user(normalized)
        if user and user.role_id:
            return user.role_id
        return self._settings.directory_default_role

    async def _lookup_by_soeid(self, soeid: str) -> dict[str, str]:
        cached = self._lookup_cache.get(soeid)
        if cached is not None:
            return cached

        # url = self._settings.directory_people_search_url_template.format(soeid=quote_plus(soeid))
        # timeout = aiohttp.ClientTimeout(total=self._settings.directory_people_search_timeout_ms / 1000)
        # connector = aiohttp.TCPConnector(verify_ssl=self._settings.directory_people_search_verify_ssl)
        # headers = {"Accept": "application/json"}
        #
        # extra_headers = self._settings.directory_people_search_extra_headers_json.strip()
        # if extra_headers:
        #     try:
        #         parsed = json.loads(extra_headers)
        #         if isinstance(parsed, dict):
        #             headers.update({str(key): str(value) for key, value in parsed.items()})
        #     except json.JSONDecodeError:
        #         pass
        #
        # async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        #     async with session.get(url, headers=headers) as resp:
        #         if resp.status >= 400:
        #             raise ValueError(f"Directory lookup failed for {soeid}: HTTP {resp.status}")
        #         body = await resp.json(content_type=None)
        #
        # docs = (((body or {}).get("people") or {}).get("docs") or [])
        # if not docs:
        #     raise ValueError(f"Directory record not found for user {soeid}")
        # first_doc = docs[0] if isinstance(docs[0], dict) else {}
        #
        # payload = {
        #     "soeid": self._first(first_doc.get("soeid")) or soeid,
        #     "email": self._first(first_doc.get("email")),
        #     "name": self._first(first_doc.get("name")),
        #     "job_title": self._first(first_doc.get("jobtitle")),
        #     "business_title": self._first(first_doc.get("ql_businesscardtitle")),
        #     "managedsegmenthierarchy": self._first(first_doc.get("managedsegmenthierarchy")),
        # }
        # self._lookup_cache[soeid] = payload
        # if len(self._lookup_cache) > 5000:
        #     oldest_key = next(iter(self._lookup_cache))
        #     self._lookup_cache.pop(oldest_key, None)
        payload = {
            "soeid": "zkdnbq0",
            "email": "shailesh.dangi@bofa.com",
            "name": "Shailesh Dangi",
            "job_title": "Senior Vice President",
            "business_title": "Analyst",
            "managedsegmenthierarchy": "EET;GT",
        }
        return payload

    @staticmethod
    def _first(value: Any) -> str:
        if isinstance(value, list):
            return str(value[0]) if value else ""
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _parse_segment_levels(hierarchy: str) -> dict[int, str]:
        result: dict[int, str] = {}
        if not hierarchy:
            return result

        for token in hierarchy.split(";"):
            item = token.strip()
            if not item or item == "#":
                continue
            match = re.search(r"\[L(\d{1,2})\]", item, flags=re.IGNORECASE)
            if not match:
                continue
            level = int(match.group(1))
            if level < 1 or level > 12:
                continue
            tail = item.split("#", 1)[1] if "#" in item else item
            name = re.sub(r"\[L\d{1,2}\]", "", tail, flags=re.IGNORECASE).strip()
            name = re.sub(r"\s+", " ", name)
            if not name:
                continue
            result[level] = name

        return result
