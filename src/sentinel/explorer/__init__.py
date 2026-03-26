"""Explorer module for signal validation and data quality diagnostics.

Provides tools for assessing whether a dataset contains meaningful
signals before investing in anomaly detection pipelines.

Key components:

- ``SignalDiagnostics``: Programmatic data quality and signal checks.
- ``Thresholds``: Configurable quality thresholds.
- ``QualityReport`` / ``CheckResult``: Structured result objects.
- ``detect_anomalies``: IQR-based anomaly detection (standalone function).
- ``detect_drift``: Statistical drift detection.
- ``load_and_label``: CSV loading with optional event labeling.
"""

from .signal_diagnostics import SignalDiagnostics, detect_anomalies
from .thresholds import Thresholds
from .report import QualityReport, CheckResult
from .drift import detect_drift
from .io import load_and_label

__all__ = [
    "SignalDiagnostics",
    "Thresholds",
    "QualityReport",
    "CheckResult",
    "detect_anomalies",
    "detect_drift",
    "load_and_label",
]
