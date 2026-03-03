"""Signal diagnostics for anomaly detection datasets.

Provides programmatic access to data quality checks without
requiring pytest as a runtime dependency.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from .report import CheckResult, QualityReport
from .thresholds import Thresholds


def detect_anomalies(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Detect anomalies in a column using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to analyze.

    Returns
    -------
    pd.DataFrame
        Rows where anomalies were detected (values outside
        ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]``).
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] < lower) | (df[column] > upper)]


class SignalDiagnostics:
    """Quick signal diagnostics for anomaly detection datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to diagnose.
    columns : list of str
        Numeric columns to analyze.
    label_column : str, optional
        Name of the binary label column (0/1). If ``None``, label-dependent
        checks are skipped.
    timestamp_column : str, optional
        Name of the timestamp column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": range(100), "b": range(100)})
    >>> diag = SignalDiagnostics(df, columns=["a", "b"])
    >>> diag.summary()  # doctest: +SKIP
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columns: List[str],
        label_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
    ):
        self.df = df
        self.columns = columns
        self.label_column = label_column
        self.timestamp_column = timestamp_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, dict]:
        """Statistical summary per column.

        Returns
        -------
        dict
            Mapping of column name to stats dict with keys:
            ``count``, ``null_pct``, ``mean``, ``std``, ``variance``,
            ``q25``, ``q50``, ``q75``, ``iqr_anomaly_count``, ``iqr_anomaly_pct``.
        """
        result: Dict[str, dict] = {}
        for col in self.columns:
            series = self.df[col]
            total = len(series)
            non_null = series.notnull().sum()
            anomalies = detect_anomalies(self.df, col)
            result[col] = {
                "count": int(non_null),
                "null_pct": round((1 - non_null / total) * 100, 2) if total else 0.0,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "variance": float(series.var()),
                "q25": float(series.quantile(0.25)),
                "q50": float(series.quantile(0.50)),
                "q75": float(series.quantile(0.75)),
                "iqr_anomaly_count": len(anomalies),
                "iqr_anomaly_pct": round(len(anomalies) / total * 100, 2) if total else 0.0,
            }
        return result

    def quality_report(self, thresholds: Optional[Thresholds] = None) -> QualityReport:
        """Run all quality checks and return a structured report.

        Parameters
        ----------
        thresholds : Thresholds, optional
            Thresholds to use. Defaults to ``Thresholds.default()``.

        Returns
        -------
        QualityReport
        """
        t = thresholds or Thresholds.default()
        checks: List[CheckResult] = []

        for col in self.columns:
            series = self.df[col]
            total = len(series)
            non_null = int(series.notnull().sum())
            non_null_pct = (non_null / total * 100) if total else 0.0
            variance = float(series.var()) if non_null > 1 else 0.0
            anomaly_pct = (len(detect_anomalies(self.df, col)) / total * 100) if total else 0.0

            checks.append(CheckResult("min_entries", non_null >= t.min_entries, non_null, t.min_entries, col))
            checks.append(CheckResult("min_non_null_pct", non_null_pct >= t.min_non_null_pct, non_null_pct, t.min_non_null_pct, col))
            checks.append(CheckResult("min_variance", variance >= t.min_variance, variance, t.min_variance, col))
            checks.append(CheckResult("anomaly_pct", anomaly_pct >= t.anomaly_pct_threshold, anomaly_pct, t.anomaly_pct_threshold, col))

        # Label-dependent checks
        if self.label_column and self.label_column in self.df.columns:
            for col in self.columns:
                corr, _ = pointbiserialr(self.df[self.label_column], self.df[col])
                checks.append(CheckResult("correlation", abs(corr) >= t.correlation_threshold, abs(corr), t.correlation_threshold, col))

        all_passed = all(c.passed for c in checks)
        return QualityReport(passed=all_passed, checks=checks)

    def correlation_report(self) -> Dict[str, dict]:
        """Compute correlations between columns and optionally with label.

        Returns
        -------
        dict
            Per-column correlation info. If ``label_column`` is set,
            includes ``point_biserial`` and ``p_value``.
        """
        result: Dict[str, dict] = {}
        for col in self.columns:
            entry: dict = {}
            if self.label_column and self.label_column in self.df.columns:
                corr, p = pointbiserialr(self.df[self.label_column], self.df[col])
                entry["point_biserial"] = round(float(corr), 4)
                entry["p_value"] = float(p)
            result[col] = entry
        return result

    def anomaly_distribution(self, method: str = "iqr") -> Dict[str, dict]:
        """Anomaly distribution per column.

        Parameters
        ----------
        method : str
            Detection method. Currently only ``"iqr"`` is supported.

        Returns
        -------
        dict
            Per-column anomaly count and percentage.
        """
        result: Dict[str, dict] = {}
        total = len(self.df)
        for col in self.columns:
            anomalies = detect_anomalies(self.df, col)
            result[col] = {
                "method": method,
                "anomaly_count": len(anomalies),
                "anomaly_pct": round(len(anomalies) / total * 100, 2) if total else 0.0,
            }
        return result

    def predictive_power(self, model=None) -> Dict[str, dict]:
        """Evaluate each feature as a single predictor of the label.

        Parameters
        ----------
        model : estimator, optional
            Scikit-learn compatible classifier. Defaults to
            ``LogisticRegression()``.

        Returns
        -------
        dict
            Per-column recall score. Empty dict if no label column.
        """
        if not self.label_column or self.label_column not in self.df.columns:
            return {}

        if model is None:
            model = LogisticRegression()

        result: Dict[str, dict] = {}
        for col in self.columns:
            X_train, X_test, y_train, y_test = train_test_split(
                self.df[[col]], self.df[self.label_column],
                test_size=0.2, random_state=42,
            )
            m = type(model)(**model.get_params())
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            rec = recall_score(y_test, y_pred, zero_division=0)
            result[col] = {"recall": round(float(rec), 4)}
        return result

    def quick_report(self, thresholds: Optional[Thresholds] = None) -> dict:
        """Run all diagnostics and return a consolidated dict.

        Parameters
        ----------
        thresholds : Thresholds, optional
            Thresholds for the quality report.

        Returns
        -------
        dict
            Keys: ``summary``, ``quality_report``, ``correlation``,
            ``anomaly_distribution``, ``predictive_power``.
        """
        return {
            "summary": self.summary(),
            "quality_report": self.quality_report(thresholds),
            "correlation": self.correlation_report(),
            "anomaly_distribution": self.anomaly_distribution(),
            "predictive_power": self.predictive_power(),
        }

    def score_distribution(self, detector, X: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores using a fitted detector.

        Parameters
        ----------
        detector : object
            A detector with ``fit`` and ``decision_function`` or ``predict`` methods.
        X : pd.DataFrame
            Data to score.

        Returns
        -------
        np.ndarray
            Anomaly scores.
        """
        detector.fit(X)
        if hasattr(detector, "decision_function"):
            return detector.decision_function(X)
        return detector.predict(X)
