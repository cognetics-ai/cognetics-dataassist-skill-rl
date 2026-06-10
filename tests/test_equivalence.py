from skillsql.verification.equivalence import result_equivalent

from _fakes import ok


def test_identical_results_equivalent():
    a = ok(["x"], [[1], [2], [3]])
    b = ok(["x"], [[1], [2], [3]])
    assert result_equivalent(a, b) is True


def test_row_order_insensitive():
    a = ok(["x"], [[1], [2], [3]])
    b = ok(["x"], [[3], [1], [2]])
    assert result_equivalent(a, b, order_insensitive=True) is True


def test_extra_predicted_columns_tolerated():
    pred = ok(["x", "extra"], [[1, "a"], [2, "b"]])
    gold = ok(["x"], [[1], [2]])
    assert result_equivalent(pred, gold, allow_extra_columns=True) is True


def test_value_mismatch_not_equivalent():
    a = ok(["x"], [[1], [2]])
    b = ok(["x"], [[1], [9]])
    assert result_equivalent(a, b) is False


def test_row_count_mismatch_not_equivalent():
    a = ok(["x"], [[1], [2], [3]])
    b = ok(["x"], [[1], [2]])
    assert result_equivalent(a, b) is False


def test_float_tolerance():
    a = ok(["x"], [[1.0000001]])
    b = ok(["x"], [[1.0000002]])
    assert result_equivalent(a, b, float_decimals=5) is True
