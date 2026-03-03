<p align="center">
  <img src="docs/images/sentinel icon with text.png" alt="Sentinel Logo" width="200"/>
</p>

# Sentinel

[![CI](https://github.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python/actions/workflows/tests.yml/badge.svg)](https://github.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python/actions/workflows/tests.yml)
[![Paper PDF](https://github.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python/actions/workflows/draft-pdf.yml/badge.svg)](https://github.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python/actions/workflows/draft-pdf.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

Signal validation and anomaly detection for enterprise log data.

Sentinel determines whether unstructured log data contains meaningful signals
before investing resources in complex anomaly detection pipelines. It provides
a modular architecture spanning ingestion, transformation, exploration,
detection, visualization, and simulation.

---

## Installation

```bash
# Base install
pip install .

# Development (includes pytest)
pip install -e ".[dev]"

# Deep learning detectors (PyTorch)
pip install ".[deep]"

# Visualization (plotly + SHAP)
pip install ".[viz]"
```

---

## Quick Start

```python
import numpy as np
import pandas as pd
from sentinel.explorer import SignalDiagnostics, Thresholds
from sentinel.detectors import IsolationForestDetector

# Generate synthetic data
rng = np.random.RandomState(42)
df = pd.DataFrame({
    "metric_a": np.concatenate([rng.normal(0, 1, 200), rng.normal(8, 0.5, 10)]),
    "metric_b": np.concatenate([rng.normal(5, 2, 200), rng.normal(20, 1, 10)]),
    "label": [0] * 200 + [1] * 10,
})

# Quick signal diagnostics
diag = SignalDiagnostics(df, columns=["metric_a", "metric_b"], label_column="label")
report = diag.quick_report(thresholds=Thresholds.relaxed())
print(report["quality_report"])

# Detect anomalies
detector = IsolationForestDetector(n_estimators=50, random_state=42)
detector.fit(df[["metric_a", "metric_b"]])
predictions = detector.predict(df[["metric_a", "metric_b"]])
print(f"Anomalies found: {(predictions == -1).sum()}")
```

---

## Modules

### Ingestion

Transforms raw log files into structured DataFrames.

| Parser | Log Format |
|--------|-----------|
| `WASParser` | WebSphere Application Server |
| `HSMParser` | Hardware Security Module |
| `HDCParser` | High-Density Computing |
| `IBMMQParser` | IBM Message Queue |
| `ZTNAParser` | Cloudflare Zero Trust Network Access |

```python
from sentinel.ingestion import LogIngestor
df = LogIngestor.ingest("path/to/logfile.log", log_type="WAS")
```

### Transformer

Rolling and string-based aggregation for time series data.

```python
from sentinel.transformer import RollingAggregator
transformer = RollingAggregator(window_size=10, aggregation_functions=["mean", "std"])
result = transformer.fit_transform(df)
```

### Explorer

Signal validation and data quality diagnostics.

```python
from sentinel.explorer import SignalDiagnostics, Thresholds
diag = SignalDiagnostics(df, columns=["col_a", "col_b"])
summary = diag.summary()
report = diag.quality_report(Thresholds.default())
```

### Detectors

| Detector | Algorithm | Dependencies |
|----------|-----------|-------------|
| `IsolationForestDetector` | Isolation Forest | scikit-learn |
| `RRCFDetector` | Robust Random Cut Forest | rrcf |
| `AutoencoderDetector` | LSTM Autoencoder | torch |
| `LNNDetector` | Liquid Neural Network | torch, ncps |

### Visualization

Static (matplotlib) and dynamic (plotly) anomaly plots, plus SHAP explanations.

### Simulation

`StreamingSimulation` for testing real-time anomaly detection scenarios.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_ingestion_quickstart` | Log parsing and custom parsers |
| `02_transformer_quickstart` | Rolling and string aggregation |
| `03_explorer_quickstart` | Signal diagnostics and quality reports |
| `04_detectors_quickstart` | Anomaly detection algorithms |
| `05_deep_detectors_quickstart` | Autoencoder and LNN detectors |
| `06_visualization_quickstart` | Anomaly and SHAP visualization |
| `07_simulation_quickstart` | Streaming simulation |
| `08_end_to_end_pipeline` | Full pipeline from ingestion to detection |

---

## Paper

The JOSS paper is in `paper/`. A draft PDF is built automatically on each push via GitHub Actions.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{sentinel2025,
  title = {Sentinel: A Python Library for Log Analysis and Anomaly Detection},
  author = {Vergara, JM and Laverde, N and Aguilar, JP and Niño, JV and Muñoz, JD and Monsalve, D and Osorio, S},
  year = {2025},
  url = {https://github.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python}
}
```
