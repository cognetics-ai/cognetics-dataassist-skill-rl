from skillsql.verification.obligations import extract_obligations, score_obligations

D = "snowflake"


def test_extract_window_rank_cue():
    obs = extract_obligations("Show the latest order per customer")
    assert "window_rank_per_group" in obs


def test_extract_period_grouping():
    obs = extract_obligations("total revenue for each month")
    assert "group_by_period" in obs or "aggregation_grain" in obs


def test_score_rewards_matching_structure():
    q = "the latest order per customer"
    good = (
        "SELECT * FROM (SELECT o.*, ROW_NUMBER() OVER "
        "(PARTITION BY customer_id ORDER BY ts DESC) rn FROM orders o) WHERE rn=1"
    )
    bad = "SELECT * FROM orders"
    sg = score_obligations(q, good, D).score
    sb = score_obligations(q, bad, D).score
    assert sg >= sb


def test_no_obligations_scores_zero():
    assert score_obligations("list all rows", "SELECT * FROM t", D).score == 0.0
