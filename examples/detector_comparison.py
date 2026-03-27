"""Compare anomaly detectors on synthetic data."""

import numpy as np
from sentinel.detectors import IsolationForestDetector


def main():
    rng = np.random.RandomState(42)
    inliers = rng.normal(0, 1, size=(200, 2))
    outliers = rng.normal(6, 0.3, size=(10, 2))
    X = np.vstack([inliers, outliers])

    # Isolation Forest
    iso = IsolationForestDetector(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X)
    preds = iso.predict(X)
    print(f"IsolationForest: {(preds == -1).sum()} anomalies detected out of {len(X)} samples")

    # Show anomaly probabilities
    proba = iso.predict_proba(X)
    top_5 = np.argsort(proba)[-5:]
    print(f"Top 5 most anomalous indices: {top_5}")
    print(f"Their probabilities: {proba[top_5].round(3)}")


if __name__ == "__main__":
    main()
