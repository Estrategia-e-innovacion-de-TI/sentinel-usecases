
"""Anomaly detection algorithms for time series data.

Available detectors:

- ``IsolationForestDetector``: Tree-based isolation.
- ``RRCFDetector``: Robust Random Cut Forest (requires ``rrcf``).
- ``AutoencoderDetector``: LSTM autoencoder (requires ``torch``).
- ``LNNDetector``: Liquid Neural Network autoencoder (requires ``torch``, ``ncps``).
- ``BaseCustomDetector``: Abstract base for user-defined detectors.
"""
from .isolation_forest import IsolationForestDetector
from .custom_detector import BaseCustomDetector


def _missing_optional_dependency(name, error):
    class _MissingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} requires optional dependencies that are not installed. "
                f"Original error: {error}"
            ) from error

    _MissingDependency.__name__ = name
    return _MissingDependency


try:
    from .autoencoder import AutoencoderDetector
except ImportError as exc:
    AutoencoderDetector = _missing_optional_dependency("AutoencoderDetector", exc)

try:
    from .rrcf_detector import RRCFDetector
except ImportError as exc:
    RRCFDetector = _missing_optional_dependency("RRCFDetector", exc)

try:
    from .lnn import LNNDetector
except ImportError as exc:
    LNNDetector = _missing_optional_dependency("LNNDetector", exc)


__all__ = [
    "AutoencoderDetector",
    "IsolationForestDetector",
    "RRCFDetector",
    "LNNDetector",
    "BaseCustomDetector",
]
