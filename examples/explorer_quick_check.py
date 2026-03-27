"""Quick data quality check using Sentinel Explorer."""

import numpy as np
import pandas as pd
from sentinel.explorer import SignalDiagnostics, Thresholds


def main():
    # Generate synthetic data with known anomalies
    rng = np.random.RandomState(42)
    n = 500
    df = pd.DataFrame({
        "cpu_usage": np.concatenate([rng.normal(50, 10, n - 20), rng.normal(95, 2, 20)]),
        "memory_mb": np.concatenate([rng.normal(2048, 256, n - 20), rng.normal(7000, 100, 20)]),
        "label": [0] * (n - 20) + [1] * 20,
    })

    # Run diagnostics
    diag = SignalDiagnostics(df, columns=["cpu_usage", "memory_mb"], label_column="label")

    print("=== Summary ===")
    for col, stats in diag.summary().items():
        print(f"\n{col}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    print("\n=== Quality Report (relaxed) ===")
    report = diag.quality_report(Thresholds.relaxed())
    print(report)
    for check in report.failed_checks:
        print(f"  {check}")

    print("\n=== Predictive Power ===")
    for col, info in diag.predictive_power().items():
        print(f"  {col}: recall={info['recall']}")


if __name__ == "__main__":
    main()
