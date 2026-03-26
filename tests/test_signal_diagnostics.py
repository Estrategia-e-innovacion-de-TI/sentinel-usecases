import numpy as np
import pandas as pd

from sentinel.explorer import SignalDiagnostics, Thresholds, detect_anomalies


def _make_df(n=200, with_label=False):
    rng = np.random.RandomState(42)
    data = {"a": rng.normal(0, 1, n), "b": rng.normal(5, 2, n)}
    if with_label:
        data["label"] = rng.choice([0, 1], n, p=[0.9, 0.1])
    return pd.DataFrame(data)


def test_detect_anomalies_basic():
    df = pd.DataFrame({"v": [1, 1, 1, 1, 1, 1, 1, 100]})
    result = detect_anomalies(df, "v")
    assert len(result) == 1
    assert result["v"].iloc[0] == 100


def test_summary_keys():
    df = _make_df()
    diag = SignalDiagnostics(df, columns=["a", "b"])
    s = diag.summary()
    assert set(s.keys()) == {"a", "b"}
    for col_stats in s.values():
        assert "count" in col_stats
        assert "variance" in col_stats
        assert "iqr_anomaly_pct" in col_stats


def test_quality_report_passes_with_relaxed():
    rng = np.random.RandomState(0)
    # Create data with clear anomalies to pass the anomaly_pct check
    normal = rng.normal(0, 1, 200)
    outliers = rng.uniform(10, 20, 20)
    values = np.concatenate([normal, outliers])
    df = pd.DataFrame({"a": values, "b": values * 2})
    diag = SignalDiagnostics(df, columns=["a", "b"])
    report = diag.quality_report(Thresholds.relaxed())
    assert report.passed is True
    assert len(report.checks) > 0


def test_quality_report_fails_with_strict():
    df = _make_df(n=50)
    diag = SignalDiagnostics(df, columns=["a", "b"])
    report = diag.quality_report(Thresholds.strict())
    assert report.passed is False
    assert len(report.failed_checks) > 0


def test_quick_report_structure():
    df = _make_df(with_label=True)
    diag = SignalDiagnostics(df, columns=["a", "b"], label_column="label")
    r = diag.quick_report()
    assert "summary" in r
    assert "quality_report" in r
    assert "correlation" in r
    assert "anomaly_distribution" in r
    assert "predictive_power" in r


def test_anomaly_distribution():
    df = _make_df()
    diag = SignalDiagnostics(df, columns=["a"])
    dist = diag.anomaly_distribution()
    assert "a" in dist
    assert "anomaly_count" in dist["a"]


def test_no_label_skips_correlation():
    df = _make_df(with_label=False)
    diag = SignalDiagnostics(df, columns=["a", "b"])
    corr = diag.correlation_report()
    assert corr["a"] == {}


def test_predictive_power_with_label():
    df = _make_df(n=200, with_label=True)
    diag = SignalDiagnostics(df, columns=["a", "b"], label_column="label")
    pp = diag.predictive_power()
    assert "a" in pp
    assert "recall" in pp["a"]
