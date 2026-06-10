"""Pluggable model layer (per-agent model selection via env)."""

from .providers import build_model, resolve_model_spec
from .registry import ResolvedModel, model_spec_for, resolve_role

__all__ = ["build_model", "resolve_model_spec", "resolve_role", "model_spec_for", "ResolvedModel"]
