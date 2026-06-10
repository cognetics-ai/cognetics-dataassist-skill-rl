"""Spider-2.0-Snow loader (proposal Section 8.1).

Parses ``spider2-snow.jsonl`` into typed tasks. Field names vary slightly across
releases, so we map the common aliases. Gold SQL is often withheld for the test
split; tasks then carry ``gold_sql=None`` and are scored by the official evaluator
offline (this loader still produces the prediction artifacts).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Spider2Task:
    instance_id: str
    db_id: str
    question: str
    gold_sql: str | None = None
    external_knowledge: str | None = None
    raw: dict | None = None


def _first(d: dict, *keys: str, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def load_spider2_snow(path: str | Path, limit: int | None = None) -> list[Spider2Task]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spider-2.0-Snow file not found: {p}")
    tasks: list[Spider2Task] = []
    for task in iter_spider2_snow(p):
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def iter_spider2_snow(path: str | Path) -> Iterator[Spider2Task]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield Spider2Task(
                instance_id=str(_first(d, "instance_id", "id", "question_id", default="")),
                db_id=str(_first(d, "db", "db_id", "database", default="")),
                question=str(_first(d, "question", "instruction", "query", default="")),
                gold_sql=_first(d, "gold_sql", "sql", "query_sql"),
                external_knowledge=_first(d, "external_knowledge", "knowledge", "evidence"),
                raw=d,
            )
