from sentinel.explorer import Thresholds


def test_default_thresholds():
    t = Thresholds.default()
    assert t.min_entries == 1000
    assert t.min_non_null_pct == 95.0


def test_strict_thresholds():
    t = Thresholds.strict()
    assert t.min_entries == 25000
    assert t.min_variance == 500.0


def test_relaxed_thresholds():
    t = Thresholds.relaxed()
    assert t.min_entries == 100
    assert t.min_non_null_pct == 80.0


def test_custom_thresholds():
    t = Thresholds(min_entries=50, min_variance=0.1)
    assert t.min_entries == 50
    assert t.min_variance == 0.1
