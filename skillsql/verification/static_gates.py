"""Decidable static-verification lattice (proposal Section 5.2).

Each gate is a *decidable* predicate over the parsed SQL (and, where needed, the
catalog). Gates are **sound for rejection**: a rejection corresponds to a
witnessed violation (forbidden node, parse failure, unresolved symbol). They are
deliberately *incomplete* -- logically-wrong-but-well-formed queries pass and are
caught by dynamic execution. False rejection of a *safe* query would corrupt the
RL gradient, so the asymmetry favors letting questionable queries through to the
execution stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
import sqlglot.expressions as exp

# Statement types permitted in the read-only Text-to-SQL setting.
_ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.With)
_FORBIDDEN = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge,
    exp.Create, exp.Drop, exp.Alter,
)


@dataclass
class GateReport:
    """Outcome of running the static lattice over one candidate query."""

    safe: bool = False
    parses: bool = False
    binds: bool = False
    scope_ok: bool = True
    join_ok: bool = True
    messages: list[str] = field(default_factory=list)
    # The first failing gate, used by the reward cascade.
    first_failure: str | None = None  # "safe" | "parse" | "bind" | None

    @property
    def passed_all(self) -> bool:
        return self.safe and self.parses and self.binds and self.scope_ok and self.join_ok


def _parse_one(sql: str, dialect: str) -> exp.Expression | None:
    statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    if len(statements) != 1:
        raise ValueError(f"expected exactly one statement, got {len(statements)}")
    return statements[0]


def gate_safe(sql: str, dialect: str) -> tuple[bool, str | None]:
    """Read-only fragment membership (a syntactic property)."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except Exception as e:  # noqa: BLE001
        return False, f"unparseable: {e}"
    if len(statements) != 1:
        return False, "multiple statements not allowed"
    root = statements[0]
    if isinstance(root, exp.Command):
        return False, f"forbidden command: {root.sql()[:40]}"
    if not isinstance(root, _ALLOWED_ROOTS):
        return False, f"non-read-only root: {type(root).__name__}"
    if any(root.find(f) for f in _FORBIDDEN):
        return False, "contains data/DDL modification"
    return True, None


def gate_parse(sql: str, dialect: str) -> tuple[bool, str | None]:
    try:
        _parse_one(sql, dialect)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"parse error: {e}"


def _collect_cte_names(tree: exp.Expression) -> set[str]:
    names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        alias = cte.alias
        if alias:
            names.add(alias.lower())
    return names


def gate_bind(
    sql: str, dialect: str, known_tables: set[str] | None
) -> tuple[bool, str | None]:
    """Resolve referenced base tables against the catalog (CTEs/aliases excluded).

    ``known_tables`` is a set of lowercase table identifiers (bare name and/or
    fully-qualified). When ``None`` (catalog unavailable), binding is skipped and
    reported as passing -- the dynamic stage will catch unresolved symbols.
    """
    if known_tables is None:
        return True, None
    try:
        tree = _parse_one(sql, dialect)
    except Exception as e:  # noqa: BLE001
        return False, f"parse error: {e}"
    cte_names = _collect_cte_names(tree)
    unknown: list[str] = []
    for tbl in tree.find_all(exp.Table):
        name = (tbl.name or "").lower()
        if not name or name in cte_names:
            continue
        fq = ".".join(
            p.lower() for p in (tbl.catalog, tbl.db, tbl.name) if p
        )
        if name not in known_tables and fq not in known_tables:
            unknown.append(fq or name)
    if unknown:
        return False, f"unresolved tables: {sorted(set(unknown))}"
    return True, None


def gate_scope(sql: str, dialect: str) -> tuple[bool, str | None]:
    """Flag window functions referenced in a bare WHERE of the same SELECT
    (a common Snowflake error; should use QUALIFY or a wrapping query)."""
    try:
        tree = _parse_one(sql, dialect)
    except Exception:  # noqa: BLE001
        return True, None  # parse handled elsewhere
    for where in tree.find_all(exp.Where):
        if where.find(exp.Window):
            return False, "window function used directly in WHERE (use QUALIFY)"
    return True, None


def gate_join(sql: str, dialect: str) -> tuple[bool, str | None]:
    """Conservative unintended-Cartesian-product detector: a JOIN with neither an
    ON predicate nor a USING clause (and not an explicit CROSS JOIN)."""
    try:
        tree = _parse_one(sql, dialect)
    except Exception:  # noqa: BLE001
        return True, None
    for join in tree.find_all(exp.Join):
        kind = (join.args.get("kind") or "").upper()
        side = (join.args.get("side") or "").upper()
        if kind == "CROSS":
            continue
        has_on = join.args.get("on") is not None
        has_using = bool(join.args.get("using"))
        if not has_on and not has_using and side in ("", "INNER"):
            return False, "join without ON/USING (possible unintended cross join)"
    return True, None


def run_static_lattice(
    sql: str, dialect: str, known_tables: set[str] | None = None
) -> GateReport:
    """Run all gates in severity order and return a consolidated report."""
    rpt = GateReport()
    rpt.safe, msg = gate_safe(sql, dialect)
    if msg:
        rpt.messages.append(f"safe: {msg}")
    if not rpt.safe:
        rpt.first_failure = "safe"
        return rpt

    rpt.parses, msg = gate_parse(sql, dialect)
    if msg:
        rpt.messages.append(f"parse: {msg}")
    if not rpt.parses:
        rpt.first_failure = "parse"
        return rpt

    rpt.binds, msg = gate_bind(sql, dialect, known_tables)
    if msg:
        rpt.messages.append(f"bind: {msg}")
    if not rpt.binds:
        rpt.first_failure = "bind"
        return rpt

    rpt.scope_ok, msg = gate_scope(sql, dialect)
    if msg:
        rpt.messages.append(f"scope: {msg}")
    rpt.join_ok, msg = gate_join(sql, dialect)
    if msg:
        rpt.messages.append(f"join: {msg}")
    return rpt
