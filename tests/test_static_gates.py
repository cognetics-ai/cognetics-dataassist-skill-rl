from skillsql.verification.static_gates import (
    gate_bind,
    gate_join,
    gate_safe,
    gate_scope,
    run_static_lattice,
)

D = "snowflake"


def test_safe_rejects_dml():
    assert gate_safe("DELETE FROM t", D)[0] is False
    assert gate_safe("UPDATE t SET x=1", D)[0] is False
    assert gate_safe("DROP TABLE t", D)[0] is False


def test_safe_rejects_multiple_statements():
    assert gate_safe("SELECT 1; SELECT 2", D)[0] is False


def test_safe_allows_select_and_cte():
    assert gate_safe("SELECT a FROM t", D)[0] is True
    assert gate_safe("WITH x AS (SELECT 1 AS a) SELECT a FROM x", D)[0] is True


def test_bind_detects_unknown_table():
    ok_, _ = gate_bind("SELECT * FROM orders", D, {"orders"})
    bad, msg = gate_bind("SELECT * FROM ghost", D, {"orders"})
    assert ok_ is True
    assert bad is False and "ghost" in msg


def test_bind_skipped_when_catalog_absent():
    assert gate_bind("SELECT * FROM anything", D, None)[0] is True


def test_bind_ignores_cte_names():
    sql = "WITH cte AS (SELECT 1 AS a) SELECT a FROM cte"
    assert gate_bind(sql, D, {"orders"})[0] is True


def test_scope_flags_window_in_where():
    sql = "SELECT a FROM t WHERE ROW_NUMBER() OVER (ORDER BY a) = 1"
    assert gate_scope(sql, D)[0] is False


def test_join_flags_missing_predicate():
    assert gate_join("SELECT * FROM a JOIN b", D)[0] is False
    assert gate_join("SELECT * FROM a CROSS JOIN b", D)[0] is True
    assert gate_join("SELECT * FROM a JOIN b ON a.id=b.id", D)[0] is True


def test_lattice_first_failure_order():
    assert run_static_lattice("DELETE FROM t", D).first_failure == "safe"
    assert run_static_lattice("SELECT * FROM ghost", D, {"orders"}).first_failure == "bind"
    good = run_static_lattice("SELECT a FROM orders WHERE a > 0", D, {"orders"})
    assert good.passed_all is True
