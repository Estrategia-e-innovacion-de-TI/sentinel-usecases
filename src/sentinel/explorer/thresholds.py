"""Configurable thresholds for data quality checks."""

from dataclasses import dataclass


@dataclass
class Thresholds:
    """Configurable thresholds for signal quality validation.

    Parameters
    ----------
    min_entries : int
        Minimum number of non-null entries per column.
    min_non_null_pct : float
        Minimum percentage of non-null values per column.
    min_variance : float
        Minimum acceptable variance per column.
    correlation_threshold : float
        Minimum absolute point-biserial correlation with label.
    recall_threshold : float
        Minimum recall for single-feature logistic regression.
    anomaly_pct_threshold : float
        Minimum percentage of IQR anomalies expected.
    """

    min_entries: int = 1000
    min_non_null_pct: float = 95.0
    min_variance: float = 0.01
    correlation_threshold: float = 0.3
    recall_threshold: float = 0.5
    anomaly_pct_threshold: float = 5.0

    @classmethod
    def default(cls) -> "Thresholds":
        """Return default thresholds."""
        return cls()

    @classmethod
    def strict(cls) -> "Thresholds":
        """Return strict thresholds for high-quality data."""
        return cls(
            min_entries=25000,
            min_non_null_pct=99.0,
            min_variance=500.0,
            correlation_threshold=0.4,
            recall_threshold=0.5,
            anomaly_pct_threshold=5.0,
        )

    @classmethod
    def relaxed(cls) -> "Thresholds":
        """Return relaxed thresholds for exploratory analysis."""
        return cls(
            min_entries=100,
            min_non_null_pct=80.0,
            min_variance=0.001,
            correlation_threshold=0.1,
            recall_threshold=0.3,
            anomaly_pct_threshold=1.0,
        )
