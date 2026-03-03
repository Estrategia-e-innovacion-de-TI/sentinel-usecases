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

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        n_fail = len(self.failed_checks)
        return f"QualityReport({status}, {len(self.checks)} checks, {n_fail} failed)"
