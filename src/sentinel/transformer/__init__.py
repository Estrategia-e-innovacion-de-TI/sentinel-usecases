"""Transformer module for time series aggregation.

Provides rolling window and string-based aggregation tools:

- ``RollingAggregator``: Rolling, expanding, or EWM window aggregations.
- ``StringAggregator``: Time-window aggregation for categorical/string data.
"""

from .rolling_aggregate import RollingAggregator
from .string_aggregate import StringAggregator

# Backward compatibility alias (typo in original name)
RollingAgregator = RollingAggregator

__all__ = ["RollingAggregator", "RollingAgregator", "StringAggregator"]
