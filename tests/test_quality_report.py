from sentinel.explorer.report import CheckResult, QualityReport


def test_check_result_repr():
    c = CheckResult("min_entries", True, 5000, 1000, "col_a")
    assert "PASS" in repr(c)

    c2 = CheckResult("min_entries", False, 50, 1000, "col_a")
    assert "FAIL" in repr(c2)


def test_quality_report_failed_checks():
    checks = [
        CheckResult("min_entries", True, 5000, 1000, "a"),
        CheckResult("min_variance", False, 0.001, 0.01, "a"),
    ]
    report = QualityReport(passed=False, checks=checks)
    assert len(report.failed_checks) == 1
    assert report.failed_checks[0].name == "min_variance"
