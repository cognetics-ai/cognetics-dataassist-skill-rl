"""Unified logging configuration for Data Assist and SkillSQL.

Default behavior is intentionally operator-friendly:
- console: colorized plain text when stdout is a terminal
- file: plain text written to ``logs/app.log`` with timed rotation

Set ``LOG_JSON=true`` or ``LOG_FORMAT=json`` to render JSON instead.
"""

from __future__ import annotations

import datetime
import json
import logging
import logging.handlers
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

_CONFIGURED = False
_CONFIG_MARKER = "_dataassist_logging_configured"
_DEFAULT_DISABLED_NOISY_MODULES = [
    "aiohttp",
    "asyncio",
    "google.adk",
    "snowflake.connector",
    "urllib3",
]
_DEFAULT_ENABLED_NOISY_MODULES: list[str] = []
_SQLALCHEMY_LOGGERS = [
    "sqlalchemy.engine",
    "sqlalchemy.engine.Engine",
]

_STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


def configure_logging(
    *,
    json: bool | None = None,
    level: int | None = None,
    force: bool = False,
) -> None:
    """Configure stdlib logging and structlog exactly once.

    ``force=True`` rebuilds handlers, which is useful for tests or commands that
    need to re-read environment variables after changing them.
    """
    global _CONFIGURED
    root_logger = logging.getLogger()

    if _CONFIGURED and not force:
        return
    if getattr(root_logger, _CONFIG_MARKER, False) and not force:
        _CONFIGURED = True
        return

    _load_root_env()
    use_json = _is_json_mode() if json is None else json
    log_level = _level_from_env() if level is None else level
    disabled_modules = _module_list_from_env(
        "DISABLE_NOISY_MODULES",
        _DEFAULT_DISABLED_NOISY_MODULES,
    )
    enabled_modules = _module_list_from_env(
        "ENABLE_NOISY_MODULES",
        _DEFAULT_ENABLED_NOISY_MODULES,
    )
    enabled_module_level = _level_from_env_var("ENABLE_NOISY_MODULES_LOG_LEVEL", logging.DEBUG)
    sqlalchemy_log_sql = _env_bool("SQLALCHEMY_LOG_SQL", default=False)
    sqlalchemy_log_level = _level_from_env_var("SQLALCHEMY_LOG_LEVEL", logging.INFO)
    if sqlalchemy_log_sql:
        enabled_modules = _unique_modules([*enabled_modules, *_SQLALCHEMY_LOGGERS])

    handler_level = log_level
    if enabled_modules:
        handler_level = min(handler_level, enabled_module_level)
    if sqlalchemy_log_sql:
        handler_level = min(handler_level, sqlalchemy_log_level)

    if force:
        reset_logging()

    shared_processors = _shared_processors()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    root_logger.addHandler(_console_handler(handler_level, use_json))
    root_logger.addHandler(_file_handler(handler_level, use_json))

    for noisy in disabled_modules:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for noisy in enabled_modules:
        logging.getLogger(noisy).setLevel(enabled_module_level)
    if sqlalchemy_log_sql:
        for logger_name in _SQLALCHEMY_LOGGERS:
            logging.getLogger(logger_name).setLevel(sqlalchemy_log_level)

    setattr(root_logger, _CONFIG_MARKER, True)
    logging.captureWarnings(True)
    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger bound to the unified configuration."""
    configure_logging()
    return structlog.get_logger(name)


def reset_logging() -> None:
    """Reset logging state and remove configured handlers."""
    global _CONFIGURED
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    if hasattr(root_logger, _CONFIG_MARKER):
        delattr(root_logger, _CONFIG_MARKER)
    structlog.reset_defaults()
    _CONFIGURED = False


def _console_handler(
    level: int,
    use_json: bool,
) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    renderer: Any
    if use_json:
        renderer = structlog.processors.JSONRenderer(serializer=_safe_json_serializer)
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=_supports_color(),
            exception_formatter=structlog.dev.plain_traceback,
        )
    handler.setFormatter(_processor_formatter(renderer))
    return handler


def _file_handler(
    level: int,
    use_json: bool,
) -> logging.Handler:
    log_path = Path(os.getenv("LOG_FILE", "logs/app.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_path,
        when=os.getenv("LOG_FILE_WHEN", "midnight"),
        interval=max(1, int(os.getenv("LOG_FILE_INTERVAL", "1"))),
        backupCount=max(0, int(os.getenv("LOG_FILE_BACKUP_COUNT", "7"))),
        encoding="utf-8",
        utc=True,
    )
    handler.setLevel(level)
    renderer: Any
    if use_json:
        renderer = structlog.processors.JSONRenderer(serializer=_safe_json_serializer)
    else:
        renderer = _plain_text_renderer
    handler.setFormatter(_processor_formatter(renderer))
    return handler


def _processor_formatter(renderer: Any) -> structlog.stdlib.ProcessorFormatter:
    return structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=[
            *_stdlib_processors(),
            _add_stdlib_extra_fields,
        ],
    )


def _shared_processors() -> list[Any]:
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        *_base_processors(),
    ]
    return processors


def _stdlib_processors() -> list[Any]:
    return [
        structlog.contextvars.merge_contextvars,
        *_base_processors(),
    ]


def _base_processors() -> list[Any]:
    return [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]


def _add_stdlib_extra_fields(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    record = event_dict.get("_record")
    if not isinstance(record, logging.LogRecord):
        return event_dict
    for key, value in record.__dict__.items():
        if key not in _STANDARD_LOG_RECORD_FIELDS and not key.startswith("_"):
            event_dict.setdefault(key, value)
    return event_dict


def _plain_text_renderer(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> str:
    event = str(event_dict.pop("event", ""))
    timestamp = str(event_dict.pop("timestamp", ""))
    level = str(event_dict.pop("level", method_name)).upper()
    logger_name = str(event_dict.pop("logger", getattr(logger, "name", "")))
    filename = event_dict.pop("filename", None)
    lineno = event_dict.pop("lineno", None)
    source = f"{logger_name}:{lineno}" if lineno else logger_name
    if filename and lineno:
        source = f"{filename}:{lineno}"

    event_dict.pop("_record", None)
    event_dict.pop("_from_structlog", None)
    exc_info = event_dict.pop("exc_info", None)
    stack = event_dict.pop("stack", None)
    exception = event_dict.pop("exception", None)

    extras = " ".join(
        f"{key}={_format_extra(value)}"
        for key, value in sorted(event_dict.items())
        if value is not None
    )
    line = f"{timestamp} {level} {source} - {event}"
    if extras:
        line = f"{line} {extras}"
    if exception:
        line = f"{line}\n{exception}"
    elif exc_info:
        line = f"{line} exc_info={_format_extra(exc_info)}"
    if stack:
        line = f"{line}\n{stack}"
    return line


def _format_extra(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, separators=(",", ":"))


def _is_json_mode() -> bool:
    log_json = os.getenv("LOG_JSON", "").lower().strip()
    log_format = os.getenv("LOG_FORMAT", "text").lower().strip()
    return log_json in {"1", "true", "yes"} or log_format == "json"


def _level_from_env() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    return getattr(logging, level_name, logging.INFO)


def _level_from_env_var(name: str, default: int) -> int:
    level_name = os.getenv(name, "").upper().strip()
    return getattr(logging, level_name, default) if level_name else default


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _module_list_from_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default)

    text = raw.strip()
    for candidate in (text, _remove_json_trailing_commas(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, str):
            return _split_module_list(parsed)
        if isinstance(parsed, list):
            return _unique_modules(str(item).strip() for item in parsed if str(item).strip())

    return _split_module_list(text)


def _split_module_list(value: str) -> list[str]:
    cleaned = value.strip().strip("[]")
    parts = re.split(r"[\n,]+", cleaned)
    modules = [part.strip().strip("\"'") for part in parts]
    return _unique_modules(module for module in modules if module)


def _remove_json_trailing_commas(value: str) -> str:
    return re.sub(r",\s*([\]}])", r"\1", value)


def _unique_modules(modules: Any) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for module in modules:
        value = str(module or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _load_root_env() -> None:
    root_env = Path(__file__).resolve().parents[2] / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _safe_json_serializer(obj: Any, **kwargs: Any) -> str:
    def default(value: Any) -> Any:
        if isinstance(value, (datetime.datetime, datetime.date)):
            return value.isoformat()
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, Exception):
            return {"type": type(value).__name__, "message": str(value)}
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return repr(value)

    kwargs.setdefault("default", default)
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, **kwargs)
