"""Compatibility wrapper for the unified app logging configuration."""

from app.observability.logging import configure_logging, get_logger, reset_logging

__all__ = ["configure_logging", "get_logger", "reset_logging"]
