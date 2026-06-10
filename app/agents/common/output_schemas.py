from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field


class RouteDecisionOutput(BaseModel):
    mode: Literal["sql", "nl"]
    reason: str = ""


class DirectoryAgentOutput(BaseModel):
    UserDirectoryInformation: dict[str, Any] = Field(default_factory=dict)
    directory_summary: dict[str, Any] = Field(default_factory=dict)


class ContextQueryExample(BaseModel):
    query_id: str = ""
    sql: str = ""
    sql2text: str = ""
    tables: list[str] = Field(default_factory=list)


class ContextBuilderOutput(BaseModel):
    role: str = ""
    business_title: str = ""
    segment_scope: list[str] = Field(default_factory=list)
    queries: list[ContextQueryExample] = Field(default_factory=list)
    examples: list[ContextQueryExample] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    table_context: list[dict[str, Any]] = Field(default_factory=list)
    similar_queries: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    metadata_summary: dict[str, Any] = Field(default_factory=dict)
    context_pack: dict[str, Any] = Field(default_factory=dict)
    context_text: str = ""
    backend_search: dict[str, Any] = Field(default_factory=dict)


class DraftPackageOutput(BaseModel):
    draft_sql: str = ""
    explanation: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CriticPackageOutput(BaseModel):
    approved: bool = False
    risk_score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    summary: str = ""


class RefinementPackageOutput(BaseModel):
    refined_sql: str = ""
    applied_recommendations: list[str] = Field(default_factory=list)
    rationale: str = ""
    exit_loop: bool = True
    exit_reason: str = ""


class ValidationPackageOutput(BaseModel):
    is_valid: bool = False
    policy_findings: list[dict[str, Any]] = Field(default_factory=list)
    explain_summary: dict[str, Any] = Field(default_factory=dict)
    risk_score: float = Field(ge=0.0, le=1.0)
    fixes: list[str] = Field(default_factory=list)


class OptimizationPackageOutput(BaseModel):
    final_sql: str = ""
    changes: list[str] = Field(default_factory=list)


class TableDescriptionOutput(BaseModel):
    table_name: str = ""
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    observed_entities: list[str] = Field(default_factory=list)
    likely_grain: str = ""
    caveats: list[str] = Field(default_factory=list)


class ColumnDescriptionItemOutput(BaseModel):
    column_name: str = ""
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    semantic_type: str = ""
    sample_values: list[str| int | float | Decimal] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ColumnDescriptionOutput(BaseModel):
    table_name: str = ""
    columns: list[ColumnDescriptionItemOutput] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class QueryToNlpOutput(BaseModel):
    query_nlp: str = ""
    tables: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


# ── SkillSQL-RL integrated outputs ─────────────────────────────────────────────

class FormalVerificationOutput(BaseModel):
    """Static-lattice gate results (Section 5.2) surfaced alongside critic/validator output."""

    safe: bool = True
    parses: bool = True
    binds: bool = True
    scope_ok: bool = True
    join_ok: bool = True
    first_failure: str | None = None        # "safe" | "parse" | "bind" | None
    gate_messages: list[str] = Field(default_factory=list)
    obligation_score: float = Field(ge=0.0, le=1.0, default=0.0)
    reward_total: float = 0.0               # composite verifier reward R(τ)
    reward_stage: str = ""                  # unsafe|parse|bind|exec_fail|matched|exec_nogold


class SkillBankContextOutput(BaseModel):
    """SkillBank skills and catalog docs injected into the context bundle."""

    skills_text: str = ""                   # formatted skills block for the generator prompt
    catalog_text: str = ""                  # top-k schema docs from pgvector
    skills_count: int = 0
    catalog_docs_count: int = 0
    source_id: str | None = None


class DistillationOutput(BaseModel):
    """Outcome of the post-workflow skill distillation step."""

    distilled: bool = False                 # True when a new skill was produced
    skill_title: str = ""
    skill_scope: str = ""                   # failure_repair | schema_specific | general_sql
    skill_id: str | None = None             # UUID of the persisted skill (if promoted)
    reason: str = ""                        # why distillation ran / was skipped
