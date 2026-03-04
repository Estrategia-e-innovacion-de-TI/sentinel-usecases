# Contributing to Sentinel

Thank you for your interest in contributing to Sentinel! We welcome contributions from the community and appreciate your help in making this project better.

## Development Setup

For detailed environment setup instructions, prerequisites, and troubleshooting, see the [Getting Started Guide](GETTINGSTARTED.md).

Quick setup:

```bash
git clone https://github.com/YOUR_USERNAME/sentinel.git
cd sentinel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Making Contributions

### Fork and Clone Workflow

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Push to your fork
6. Submit a pull request

### Branch Naming Conventions

Use descriptive branch names that indicate the purpose of your changes:
- `feature/your-feature-name` for new features
- `fix/bug-description` for bug fixes
- `docs/documentation-update` for documentation changes

### Commit Message Standard

Commit messages must follow this structure:

```
<type>(<scope>)!: <description>
```

Where:

- `<type>` (required): Indicates the category of the change. Allowed values:

  | Type | Description |
  |------|-------------|
  | `feat` | New feature |
  | `fix` | Bug fix |
  | `docs` | Documentation changes |
  | `style` | Formatting changes (spaces, commas, etc.) with no code impact |
  | `refactor` | Code restructuring without changing functionality |
  | `perf` | Performance improvements |
  | `test` | Adding or modifying tests |
  | `build` | Changes to the build system or dependencies |
  | `ci` | Continuous integration configuration |
  | `removed` | Deleted code |
  | `deprecated` | Code marked as obsolete |
  | `security` | Security-related changes |
  | `chore` | Maintenance tasks or minor refactoring |

  Release-specific types:

  | Type | Description |
  |------|-------------|
  | `featurerelease` | New feature release |
  | `securitypatchrelease` | Security patch |
  | `fixpatchrelease` | Bug fix patch |
  | `breakingrelease` | Release with breaking changes |
  | `breaking` | Incompatible / breaking change |

- `(<scope>)` (optional): Specifies which part of the codebase is affected. Can be a module, file, or specific component (e.g., `ingestion`, `detectors`, `explorer`).

- `!` (optional): Indicates the change is significant or has a major impact.

- `<description>` (required): A brief description of the change.

Examples:

```
feat(detectors): add RRCFDetector streaming support
fix(transformer): resolve KeyError in StringAggregator groupby
docs: update README with new visualization methods
refactor(explorer)!: redesign QualityReport API
ci: add Python 3.12 to test matrix
```

### Pull Request Process

1. Ensure your code passes all tests
2. Update documentation if you've changed functionality
3. Describe your changes clearly in the pull request description
4. Link any related issues
5. Wait for review from maintainers

## Issue Labeling

Labels on GitHub Issues are a fundamental part of efficient project management. They enable clear, practical communication and make issue administration more transparent for all contributors interacting with the repository.

We classify issues to improve visibility and help contributors quickly identify the type of work involved (bug, vulnerability, documentation, etc.).

> Labels are grouped by a prefix and an associated color for easy identification.

### Category Labels

Prefix `c:` (green) — classifies issues by their nature:

| Label | Description |
|-------|-------------|
| `c: documentation` | Related to documentation |
| `c: feature` | Related to a new feature or enhancement |
| `c: vulnerability` | Related to a detected vulnerability |
| `c: bug` | Related to a bug or error |

### General Labels

Prefix `g:` (light blue) — generic labels for organizing and tracking:

| Label | Description |
|-------|-------------|
| `g: good first issue` | A good entry point for first-time contributors |
| `g: help wanted` | Needs someone to be assigned |
| `g: in triage` | Under initial analysis before assignment |
| `g: assigned for triage` | Escalated to an expert for deeper analysis before assignment |
| `g: question` | The issue is more of a question than a problem |

### Resolution Labels

Prefix `r:` (red) — closure labels to finalize an issue:

| Label | Description |
|-------|-------------|
| `r: duplicate` | Closed because another issue reports the same problem |
| `r: fixed` | Closed because the fix is being addressed in another issue |
| `r: solved` | Closed because the solution was implemented from another issue |
| `r: invalid` | Closed because the issue is invalid |
| `r: timeout` | Closed because the author did not provide details within the expected time |
| `r: wontfix` | Closed because it will not be fixed |

## Triage Process

Triage is the initial state of every issue or report created on GitHub. It is the first filter before a collaborator addresses it. The initial validation reviews the arguments provided, assesses urgency, and then assigns a classification along with the appropriate descriptive labels.

The triage process as the first step for every report allows us to filter out poorly created reports, missing arguments, duplicates, and critical/urgent items, so they can then receive proper attention.

At the community or project level, this first control over issues enables solid and transparent ticket management. It helps prioritize and respond to requests according to their urgency, based on the classification assigned during triage.

> Triage analysis is performed by community members who hold the **Maintainer** role.

## Code Style

- Follow PEP 8 style guidelines for Python code
- Write clear, readable code with meaningful variable names
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write tests for new functionality

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## Questions?

If you have questions or need help, feel free to:
- Open an issue on GitHub
- Reach out to the maintainers

We appreciate your contributions and look forward to working with you!
