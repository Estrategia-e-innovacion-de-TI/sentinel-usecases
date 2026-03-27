"""Example: rolling aggregation on time series data."""

import numpy as np
import pandas as pd
from sentinel.transformer import RollingAggregator


def main():
    # Create synthetic time series
    rng = np.random.RandomState(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "value": np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.2, 100),
    })

    # Apply rolling aggregation
    transformer = RollingAggregator(
        window_size=10,
        aggregation_functions=["mean", "std"],
        columns="value",
    )
    result = transformer.fit_transform(df)

    print("Original columns:", list(df.columns))
    print("After transform:", list(result.columns))
    print(result[["value", "value_rolling_mean", "value_rolling_std"]].tail(10))


if __name__ == "__main__":
    main()
