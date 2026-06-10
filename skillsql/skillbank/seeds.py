"""Curated seed skills for the SqlSkillBank (proposal Section 7, Table 1).

These are the ``general_sql`` and ``dialect`` (Snowflake) skills that are *always*
injected into the policy's context -- they encode SQL fundamentals and
Snowflake-specific idioms that apply across all tasks. They are the starting point
before distillation produces schema-specific and failure-repair skills.

Each skill dict maps directly to the ``Skill`` ORM model fields. Call
:func:`load_seeds` to insert them into the catalog (idempotent -- existing titles
are skipped to avoid overwriting distilled skills).
"""

from __future__ import annotations

from typing import Any

# ── General SQL skills (scope = "general_sql") ────────────────────────────────
GENERAL_SQL_SEEDS: list[dict[str, Any]] = [
    {
        "scope": "general_sql",
        "skill_type": "strategy",
        "title": "Declare output grain before aggregating",
        "principle": (
            "Before writing any aggregation function (SUM, COUNT, AVG, …), decide the "
            "grain of the output -- one row per what? -- and ensure the GROUP BY columns "
            "match the non-aggregated SELECT columns exactly. Missing a GROUP BY column "
            "is the leading cause of wrong-grain results."
        ),
        "when_to_apply": "Any question involving totals, averages, counts, or summaries.",
        "positive_example": (
            "-- Correct: grain is (customer_id, month)\n"
            "SELECT customer_id, DATE_TRUNC('month', order_date) AS month,\n"
            "       SUM(amount) AS total\n"
            "FROM orders\n"
            "GROUP BY customer_id, DATE_TRUNC('month', order_date)"
        ),
        "negative_example": (
            "-- Wrong: customer_name not in GROUP BY\n"
            "SELECT customer_id, customer_name, SUM(amount)\n"
            "FROM orders\n"
            "GROUP BY customer_id"
        ),
        "status": "promoted",
    },
    {
        "scope": "general_sql",
        "skill_type": "strategy",
        "title": "Build a date/entity spine for activity-absent periods",
        "principle": (
            "When the question asks for results across a calendar period including "
            "time slots with no activity (zero sales, absent employees, …), generate "
            "a date or entity spine first, then LEFT JOIN the fact table. A plain "
            "GROUP BY will silently omit empty periods."
        ),
        "when_to_apply": (
            "Question contains: 'per month with no activity', 'for each day even if zero', "
            "'months without orders', 'all days in range'."
        ),
        "positive_example": (
            "WITH months AS (\n"
            "  SELECT DATEADD(month, seq4(), '2024-01-01')::DATE AS month\n"
            "  FROM TABLE(GENERATOR(ROWCOUNT => 12))\n"
            ")\n"
            "SELECT m.month, COALESCE(SUM(o.amount), 0) AS revenue\n"
            "FROM months m\n"
            "LEFT JOIN orders o ON DATE_TRUNC('month', o.order_date) = m.month\n"
            "GROUP BY m.month"
        ),
        "negative_example": (
            "-- Silently drops months with no orders\n"
            "SELECT DATE_TRUNC('month', order_date), SUM(amount)\n"
            "FROM orders\n"
            "GROUP BY 1"
        ),
        "status": "promoted",
    },
    {
        "scope": "general_sql",
        "skill_type": "strategy",
        "title": "Use window ranking for top-N per group",
        "principle": (
            "To select the top/latest/earliest N records per partition (e.g., latest order "
            "per customer), use a window function (ROW_NUMBER, RANK, DENSE_RANK) with "
            "PARTITION BY and ORDER BY inside an OVER clause, then filter by the rank. "
            "Never use a correlated subquery with MAX/MIN for this -- it is slower and "
            "breaks on ties."
        ),
        "when_to_apply": (
            "Question asks for 'latest', 'most recent', 'top N per', 'first/last per group'."
        ),
        "positive_example": (
            "SELECT * FROM (\n"
            "  SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn\n"
            "  FROM orders\n"
            ") ranked\n"
            "WHERE rn = 1  -- or use QUALIFY in Snowflake"
        ),
        "negative_example": (
            "-- Brittle and slow on large tables\n"
            "SELECT * FROM orders o\n"
            "WHERE order_date = (SELECT MAX(order_date) FROM orders WHERE customer_id = o.customer_id)"
        ),
        "status": "promoted",
    },
    {
        "scope": "general_sql",
        "skill_type": "strategy",
        "title": "Handle NULLs explicitly in filters and comparisons",
        "principle": (
            "NULL comparisons with = or != always return NULL (not TRUE/FALSE). Use "
            "IS NULL / IS NOT NULL for null checks. For aggregations, remember that "
            "COUNT(*) counts NULLs but COUNT(col) does not. Use COALESCE or NULLIF "
            "to handle null-as-zero and division-by-zero cases."
        ),
        "when_to_apply": "Any filter, join condition, or calculation involving nullable columns.",
        "positive_example": (
            "-- Correct null handling\n"
            "SELECT customer_id,\n"
            "       COALESCE(SUM(amount), 0) AS total,\n"
            "       NULLIF(COUNT(CASE WHEN status = 'refund' THEN 1 END), 0) AS refunds\n"
            "FROM orders\n"
            "WHERE cancelled_at IS NULL\n"
            "GROUP BY customer_id"
        ),
        "negative_example": (
            "-- Wrong: NULL = NULL is NULL, not TRUE\n"
            "WHERE cancelled_at = NULL"
        ),
        "status": "promoted",
    },
    {
        "scope": "general_sql",
        "skill_type": "strategy",
        "title": "Use CTEs for multi-step queries; alias CTE outputs cleanly",
        "principle": (
            "Break complex queries into named CTEs (WITH clauses) rather than deeply "
            "nested subqueries. Give each CTE a clear, lowercase_snake_case name. "
            "Downstream CTEs must reference CTE *output* column aliases, not the inner "
            "source table columns -- this is the most common 'invalid identifier' error "
            "in multi-step queries."
        ),
        "when_to_apply": "Any query with more than two logical steps or requiring intermediate aggregations.",
        "positive_example": (
            "WITH monthly_revenue AS (\n"
            "  SELECT customer_id, DATE_TRUNC('month', order_date) AS month,\n"
            "         SUM(amount) AS revenue\n"
            "  FROM orders GROUP BY 1, 2\n"
            "),\n"
            "ranked AS (\n"
            "  SELECT *, RANK() OVER (PARTITION BY month ORDER BY revenue DESC) AS rnk\n"
            "  FROM monthly_revenue  -- references 'revenue' not 'SUM(amount)'\n"
            ")\n"
            "SELECT * FROM ranked WHERE rnk <= 3"
        ),
        "negative_example": (
            "-- Wrong: referencing 'SUM(amount)' from inner query\n"
            "WITH monthly_revenue AS (...)\n"
            "SELECT customer_id, RANK() OVER (ORDER BY SUM(amount) DESC)\n"
            "FROM monthly_revenue"
        ),
        "status": "promoted",
    },
]

# ── Snowflake dialect skills (scope = "dialect", dialect = "snowflake") ────────
SNOWFLAKE_DIALECT_SEEDS: list[dict[str, Any]] = [
    {
        "scope": "dialect",
        "skill_type": "dialect_heuristic",
        "dialect": "snowflake",
        "title": "Use QUALIFY to filter window function results",
        "principle": (
            "In Snowflake, filter the result of a window function using QUALIFY instead "
            "of wrapping the query in a subquery. QUALIFY operates after the window "
            "function is computed. A bare WHERE clause that references a window function "
            "alias will raise a compilation error."
        ),
        "when_to_apply": (
            "Any time you need to filter by ROW_NUMBER(), RANK(), DENSE_RANK(), "
            "or any other window function result."
        ),
        "positive_example": (
            "SELECT customer_id, order_date, amount\n"
            "FROM orders\n"
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1"
        ),
        "negative_example": (
            "-- Compilation error in Snowflake\n"
            "SELECT customer_id, order_date, amount,\n"
            "       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn\n"
            "FROM orders\n"
            "WHERE rn = 1  -- rn is not yet defined in WHERE"
        ),
        "status": "promoted",
    },
    {
        "scope": "dialect",
        "skill_type": "dialect_heuristic",
        "dialect": "snowflake",
        "title": "Generate date/time series with TABLE(GENERATOR())",
        "principle": (
            "Snowflake does not have a built-in generate_series(). Use "
            "TABLE(GENERATOR(ROWCOUNT => n)) with SEQ4() or DATEADD to produce integer "
            "or date sequences. For date spines, combine with DATEADD(day/month, SEQ4(), start_date)."
        ),
        "when_to_apply": "Building date spines, calendar tables, or sequential ID ranges.",
        "positive_example": (
            "-- 365-day spine starting 2024-01-01\n"
            "SELECT DATEADD(day, SEQ4(), '2024-01-01'::DATE) AS day\n"
            "FROM TABLE(GENERATOR(ROWCOUNT => 365))"
        ),
        "negative_example": (
            "-- generate_series does not exist in Snowflake\n"
            "SELECT * FROM generate_series('2024-01-01'::DATE, '2024-12-31'::DATE, INTERVAL '1 day')"
        ),
        "status": "promoted",
    },
    {
        "scope": "dialect",
        "skill_type": "dialect_heuristic",
        "dialect": "snowflake",
        "title": "Parse semi-structured data with FLATTEN and PARSE_JSON",
        "principle": (
            "To query JSON/VARIANT columns, use PARSE_JSON(col):key for simple paths or "
            "LATERAL FLATTEN(INPUT => col, PATH => 'key') to unnest arrays. Use the "
            "::<type> cast (e.g., ::STRING, ::NUMBER) after extraction because VARIANT "
            "lookups return VARIANT, not typed values."
        ),
        "when_to_apply": "Columns stored as VARIANT, OBJECT, or ARRAY; JSON extraction.",
        "positive_example": (
            "SELECT f.value:product_id::STRING AS product_id,\n"
            "       f.value:quantity::NUMBER    AS qty\n"
            "FROM orders,\n"
            "LATERAL FLATTEN(INPUT => order_items) f"
        ),
        "negative_example": (
            "-- Missing cast; downstream comparison fails\n"
            "SELECT order_items:product_id FROM orders WHERE order_items:quantity > 1"
        ),
        "status": "promoted",
    },
    {
        "scope": "dialect",
        "skill_type": "dialect_heuristic",
        "dialect": "snowflake",
        "title": "Use DATE_TRUNC for period grouping; avoid TO_CHAR for aggregation",
        "principle": (
            "For aggregating by calendar period, use DATE_TRUNC('month'/'quarter'/'year', date_col) "
            "which preserves the DATE type and sorts correctly. Avoid TO_CHAR() for grouping -- "
            "it produces a string which breaks date ordering and wastes the optimizer."
        ),
        "when_to_apply": "Monthly/quarterly/annual aggregations, time-series GROUP BY.",
        "positive_example": (
            "SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount)\n"
            "FROM orders\n"
            "GROUP BY DATE_TRUNC('month', order_date)\n"
            "ORDER BY month"
        ),
        "negative_example": (
            "-- String sort order breaks in month names like 'Apr', 'Aug', 'Dec'...\n"
            "SELECT TO_CHAR(order_date, 'Mon YYYY') AS month, SUM(amount)\n"
            "FROM orders\n"
            "GROUP BY TO_CHAR(order_date, 'Mon YYYY')"
        ),
        "status": "promoted",
    },
    {
        "scope": "dialect",
        "skill_type": "dialect_heuristic",
        "dialect": "snowflake",
        "title": "Use ILIKE for case-insensitive string matching",
        "principle": (
            "Snowflake string comparisons with LIKE are case-sensitive by default. "
            "Use ILIKE for case-insensitive matching instead of UPPER(col) LIKE UPPER(pattern). "
            "For multi-pattern matching, use REGEXP_ILIKE."
        ),
        "when_to_apply": "String pattern matching, name lookups, free-text filters.",
        "positive_example": (
            "SELECT * FROM customers WHERE email ILIKE '%@example.com'"
        ),
        "negative_example": (
            "-- Verbose and slower\n"
            "SELECT * FROM customers WHERE UPPER(email) LIKE UPPER('%@EXAMPLE.COM')"
        ),
        "status": "promoted",
    },
]

# ── Verifier obligation skills (scope = "verifier_obligation") ─────────────────
VERIFIER_OBLIGATION_SEEDS: list[dict[str, Any]] = [
    {
        "scope": "verifier_obligation",
        "skill_type": "obligation",
        "title": "Obligation: months/periods with no activity require a spine",
        "principle": (
            "If the question asks for results per calendar period and explicitly mentions "
            "periods with no activity (zero, absent, none), you MUST build a period spine "
            "and LEFT JOIN facts -- not just GROUP BY on the fact table."
        ),
        "when_to_apply": "Question mentions 'per month with no activity', 'even if zero', 'all periods'.",
        "status": "promoted",
    },
    {
        "scope": "verifier_obligation",
        "skill_type": "obligation",
        "title": "Obligation: top/latest per group requires window function",
        "principle": (
            "If the question asks for the top/latest/first N records per group, "
            "you MUST use a window function (ROW_NUMBER/RANK) with PARTITION BY, "
            "not a plain ORDER BY + LIMIT (which ignores the partition boundary)."
        ),
        "when_to_apply": "Question asks for latest, first, or top N per distinct entity.",
        "status": "promoted",
    },
]

# ── All seeds ──────────────────────────────────────────────────────────────────
ALL_SEEDS: list[dict[str, Any]] = (
    GENERAL_SQL_SEEDS + SNOWFLAKE_DIALECT_SEEDS + VERIFIER_OBLIGATION_SEEDS
)


def load_seeds(repo: "CatalogRepository") -> int:  # type: ignore[name-defined]  # noqa: F821
    """Insert seed skills that do not yet exist (matched by title). Returns count inserted."""
    from ..catalog.models import Skill

    inserted = 0
    with repo.session() as s:
        for seed in ALL_SEEDS:
            if s.query(Skill).filter_by(title=seed["title"]).first():
                continue
            s.add(Skill(**seed))
            inserted += 1
        if inserted:
            s.commit()
    return inserted
