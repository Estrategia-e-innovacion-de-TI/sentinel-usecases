"""Structured result dataclasses for quality reports."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CheckResult:
    """Result of a single quality check.

    Parameters
    ----------
    name : str
        Name of the check (e.g. ``"min_entries"``).
    passed : bool
        Whether the check passed.
    value : float
        Observed value.
    threshold : float
        Threshold used for comparison.
    column : str
        Column the check was applied to.
    """

    name: str
    passed: bool
    value: float
    threshold: float
    column: str

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} | {self.column}: {self.value:.4f} (threshold={self.threshold})"


@dataclass
class QualityReport:
    """Aggregated quality report containing multiple check results.

    Parameters
    ----------
    passed : bool
        Whether all checks passed.
    checks : list of CheckResult
        Individual check results.
    """

    passed: bool
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def failed_checks(self) -> List[CheckResult]:
        """Return only the checks that failed."""
        return [c for c in self.checks if not c.passed]

    @property
    def score(self) -> float:
        """Fraction of checks that passed (0.0 to 1.0)."""
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    def interpret(self) -> str:
        """Generate a human-readable interpretation of the report.

        Analyzes which checks failed, explains what each failure means,
        and provides actionable recommendations.

        Returns
        -------
        str
            Multi-line interpretation text.
        """
        lines: List[str] = []
        n_total = len(self.checks)
        n_passed = sum(1 for c in self.checks if c.passed)
        n_failed = n_total - n_passed

        # Overall verdict
        if self.passed:
            lines.append(
                f"SIGNAL DETECTED — All {n_total} checks passed "
                f"(score: {self.score:.0%})."
            )
            lines.append(
                "The data has sufficient volume, variance, and anomaly "
                "presence to proceed with anomaly detection."
            )
            return "\n".join(lines)

        lines.append(
            f"INSUFFICIENT SIGNAL — {n_failed}/{n_total} checks failed "
            f"(score: {self.score:.0%})."
        )

        # Group failures by check name
        failure_groups: dict = {}
        for c in self.failed_checks:
            failure_groups.setdefault(c.name, []).append(c)

        _explanations = {
            "min_entries": (
                "Not enough data points. The dataset has fewer rows than "
                "the required minimum. Collect more data or use "
                "Thresholds.relaxed() for exploratory analysis."
            ),
            "min_non_null_pct": (
                "Too many missing values. Consider imputation, dropping "
                "sparse columns, or investigating the data source."
            ),
            "min_variance": (
                "Near-constant signal. The column has very low variance, "
                "meaning there is little variation for a detector to learn "
                "from. Check if the column is informative or if the data "
                "needs a different time range."
            ),
            "anomaly_pct": (
                "Too few IQR outliers detected. The data distribution may "
                "be too uniform, or the anomalies are subtle. Consider "
                "domain-specific feature engineering or a different "
                "detection method."
            ),
            "correlation": (
                "Weak correlation between features and the label. The "
                "features may not be predictive of the anomaly class. "
                "Consider adding more informative features or reviewing "
                "the labeling criteria."
            ),
        }

        for name, checks in failure_groups.items():
            cols = ", ".join(c.column for c in checks)
            explanation = _explanations.get(name, "Check failed.")
            lines.append("")
            lines.append(f"  [{name}] Failed for: {cols}")
            lines.append(f"    → {explanation}")

        # Recommendation
        lines.append("")
        if "min_entries" in failure_groups and len(failure_groups) == 1:
            lines.append(
                "Recommendation: The only issue is data volume. If this is "
                "a sample, try with the full dataset. Otherwise, use "
                "Thresholds.relaxed() to proceed with exploratory analysis."
            )
        elif "min_variance" in failure_groups:
            lines.append(
                "Recommendation: Review whether low-variance columns carry "
                "useful information. Drop or transform them before detection."
            )
        else:
            lines.append(
                "Recommendation: Address the failed checks above before "
                "investing in anomaly detection pipelines. Use "
                "Thresholds.relaxed() for a less strict assessment."
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        n_fail = len(self.failed_checks)
        return f"QualityReport({status}, {len(self.checks)} checks, {n_fail} failed)"
