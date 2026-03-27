# Getting Started

This guide walks you through setting up a local development environment for Sentinel.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [Python](https://www.python.org/) | ≥ 3.10 | Runtime (project tested on 3.12) |
| [Git](https://git-scm.com/) | latest | Version control |
| [pip](https://pip.pypa.io/) | latest | Package installer (ships with Python) |

Optional but recommended:

| Tool | Purpose |
|------|---------|
| [uv](https://docs.astral.sh/uv/) | Fast Python package manager and virtualenv tool |
| [Jupyter](https://jupyter.org/) / [VS Code](https://code.visualstudio.com/) | Running the quickstart notebooks |

## Step-by-Step Installation

### 1. Clone the repository

```bash
git clone https://github.com/bancolombia/sentinel.git
cd sentinel
```

### 2. Create a virtual environment

Using `venv` (built-in):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

Or using `uv`:

```bash
uv venv .venv
source .venv/bin/activate
```

### 3. Install the package

Choose the install profile that fits your needs:

```bash
# Base install — ingestion, transformer, explorer, IsolationForest detector
pip install -e .

# Development — adds pytest, ipykernel, nbformat
pip install -e ".[dev]"

# Deep learning detectors — AutoencoderDetector, LNNDetector (PyTorch + ncps)
pip install -e ".[deep]"

# Visualization — AnomalyVisualizer (Plotly), SHAPVisualizer
pip install -e ".[viz]"

# Robust Random Cut Forest detector
pip install -e ".[rrcf]"

# Everything at once
pip install -e ".[all]"
```

### 4. Verify the installation

```bash
python -c "import sentinel; print('Sentinel installed successfully')"
```

### 5. Run the test suite

```bash
pytest -q
```

All 17 tests should pass.

### 6. Explore the notebooks

Launch Jupyter and open any notebook in the `notebooks/` folder:

```bash
jupyter notebook notebooks/
```

| # | Notebook | What you will learn |
|---|----------|---------------------|
| 01 | Ingestion Quickstart | Log parsing and custom parsers |
| 02 | Transformer Quickstart | Rolling and string aggregation |
| 03 | Explorer Quickstart | Signal diagnostics and drift detection |
| 04 | Detectors Quickstart | IsolationForest and RRCF |
| 05 | Deep Detectors Quickstart | Autoencoder and Liquid Neural Networks |
| 06 | Visualization Quickstart | Static/interactive plots and SHAP |
| 07 | Simulation Quickstart | Streaming anomaly detection |
| 08 | End-to-End Pipeline | Full workflow from ingestion to SHAP |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'pkg_resources'` | Make sure `setuptools<81` is installed (`pip install "setuptools<81"`). The `rrcf` package depends on it. |
| `ImportError` for torch / ncps | Install the deep extras: `pip install -e ".[deep]"` |
| `ImportError` for plotly / shap | Install the viz extras: `pip install -e ".[viz]"` |
| Tests fail with import errors | Run `pip install -e ".[all]"` to install all optional dependencies |

## Next Steps

- Read the [CONTRIBUTING.md](CONTRIBUTING.md) guide to learn about the development workflow, commit standards, and issue labeling.
- Check the [README.md](README.md) for module documentation and code examples.
- Browse the `examples/` folder for standalone scripts.
