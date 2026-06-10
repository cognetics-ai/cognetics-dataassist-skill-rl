import pytest
from skillsql.verification.reward import RewardConfig, compute_reward

from _fakes import FakeConnector, ok

Q = "total revenue per month"
KNOWN = {"orders"}


@pytest.mark.asyncio
async def test_unsafe_query_floored():
    conn = FakeConnector()
    rb = await compute_reward(question=Q, sql="DELETE FROM orders", connector=conn, known_tables=KNOWN)
    assert rb.stage == "unsafe" and rb.total == -1.0


@pytest.mark.asyncio
async def test_bind_failure_penalty():
    conn = FakeConnector()
    rb = await compute_reward(question=Q, sql="SELECT * FROM ghost", connector=conn, known_tables=KNOWN)
    assert rb.stage == "bind" and rb.total == RewardConfig().penalty_bind


@pytest.mark.asyncio
async def test_execution_failure_penalty():
    sql = "SELECT x FROM orders"
    conn = FakeConnector(results={sql: ok([], []).model_copy(update={"error": "boom"})})
    rb = await compute_reward(question=Q, sql=sql, connector=conn, known_tables=KNOWN)
    assert rb.stage == "exec_fail" and rb.total < 0


@pytest.mark.asyncio
async def test_correct_match_dominates():
    sql = "SELECT month, SUM(amount) FROM orders GROUP BY month"
    res = ok(["month", "rev"], [["2024-01", 10], ["2024-02", 20]])
    conn = FakeConnector(results={sql: res})
    rb = await compute_reward(question=Q, sql=sql, connector=conn, gold=res, known_tables=KNOWN)
    assert rb.equivalent is True
    assert rb.total >= 1.0  # exact-match dominance


@pytest.mark.asyncio
async def test_exec_nogold_bounded_positive():
    sql = "SELECT month, SUM(amount) FROM orders GROUP BY month"
    res = ok(["month", "rev"], [["2024-01", 10]])
    conn = FakeConnector(results={sql: res})
    rb = await compute_reward(question=Q, sql=sql, connector=conn, gold=None, known_tables=KNOWN)
    assert rb.stage == "exec_nogold" and 0.0 < rb.total < 1.0
