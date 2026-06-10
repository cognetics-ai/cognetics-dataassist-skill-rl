from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class User:
    soeid: str
    display_name: str
    email: str
    role_id: str


@dataclass
class UsersWorkInformation:
    job_profile_title: str
    job_family_group: str
    title: str
    business_title: str
    managed_segment_path: str
    managed_geograph_path: str

    def as_directory_payload(self) -> dict[str, str]:
        return {
            "JobProfileTitle": self.job_profile_title,
            "JobFamilyGroup": self.job_family_group,
            "Title": self.title,
            "BusinessTitle": self.business_title,
            "ManagedSegmentPath": self.managed_segment_path,
            "ManagedGeographPath": self.managed_geograph_path,
        }


@dataclass
class UserDirectoryInformation:
    users_id: str
    role_id: str
    work_information: UsersWorkInformation

    def as_payload(self) -> dict[str, dict]:
        return {
            "UserDirectoryInformation": {
                "UsersID": self.users_id,
                "UsersWorkInformation": self.work_information.as_directory_payload(),
            }
        }


@dataclass
class QueryHistoryEntry:
    query_id: str
    soeid: str
    role_id: str
    engine: str
    sql_text: str
    sql2text: str
    tables: list[str]
    created_at: datetime = field(default_factory=utcnow)
    status: str = "succeeded"


@dataclass
class QueryRun:
    run_id: str
    soeid: str
    engine: str
    submitted_text: str
    input_mode: str = "auto"
    route_mode: str | None = None
    submitted_sql: str | None = None
    submitted_prompt: str | None = None
    natural_language_query: str | None = None
    embedding: list[float] | None = None
    source_id: str | None = None
    reward_json: dict[str, Any] | None = None
    final_sql: str = ""
    status: str = "queued"
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error_message: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    schema: list[dict[str, Any]] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)


@dataclass
class RunEvent:
    run_id: str
    event_type: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=utcnow)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_sse(self) -> str:
        data = {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }
        import json

        return f"data: {json.dumps(data)}\n\n"
