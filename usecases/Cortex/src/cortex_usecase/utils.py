"""Utility helpers for the Cortex management audit log use case."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


UTC = timezone.utc


@dataclass(frozen=True)
class UseCasePaths:
    """Resolved paths used by the Cortex use case."""

    repo_root: Path
    usecase_root: Path
    notebooks_dir: Path
    src_dir: Path
    raw_dir: Path
    processed_dir: Path
    outputs_dir: Path
    figures_dir: Path

    def ensure(self) -> "UseCasePaths":
        """Create the directory tree if it does not already exist."""
        for directory in (
            self.usecase_root,
            self.notebooks_dir,
            self.src_dir,
            self.raw_dir,
            self.processed_dir,
            self.outputs_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate the repository root by walking upward until ``pyproject.toml`` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not resolve the Sentinel repository root from the current path.")


def resolve_usecase_paths(repo_root: Optional[Path] = None) -> UseCasePaths:
    """Resolve and create the directory tree for the Cortex use case."""
    repo = find_repo_root(repo_root)
    usecase_root = repo / "usecases" / "Cortex"
    return UseCasePaths(
        repo_root=repo,
        usecase_root=usecase_root,
        notebooks_dir=usecase_root / "notebooks",
        src_dir=usecase_root / "src",
        raw_dir=usecase_root / "data" / "raw",
        processed_dir=usecase_root / "data" / "processed",
        outputs_dir=usecase_root / "outputs",
        figures_dir=usecase_root / "outputs" / "figures",
    ).ensure()


def parse_datetime(value: Any) -> Optional[datetime]:
    """Parse an ISO-like datetime value and normalize it to UTC."""
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def resolve_time_window(
    lookback_hours: int = 6,
    start_time: Optional[Any] = None,
    end_time: Optional[Any] = None,
    now: Optional[datetime] = None,
) -> tuple[datetime, datetime]:
    """Resolve a UTC start/end window from explicit dates or a lookback."""
    reference_now = parse_datetime(now) or datetime.now(UTC)
    resolved_end = parse_datetime(end_time) or reference_now
    resolved_start = parse_datetime(start_time) or (resolved_end - timedelta(hours=lookback_hours))

    if resolved_start >= resolved_end:
        raise ValueError("The extraction start time must be earlier than the end time.")

    return resolved_start, resolved_end


def to_epoch_millis(value: Any) -> int:
    """Convert a datetime-like value to epoch milliseconds."""
    parsed = parse_datetime(value)
    if parsed is None:
        raise ValueError("A datetime value is required to compute epoch milliseconds.")
    return int(parsed.timestamp() * 1000)


def make_extraction_id(
    start_time: datetime,
    end_time: datetime,
    prefix: str = "cortex_mgmt_audit",
) -> str:
    """Create a stable identifier for persisted extraction artifacts."""
    start_text = parse_datetime(start_time).strftime("%Y%m%dT%H%M%SZ")
    end_text = parse_datetime(end_time).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{start_text}_{end_text}"


def sanitize_category_name(value: Any, max_length: int = 48) -> str:
    """Convert a category value into a filesystem and column-safe token."""
    text = str(value).strip().lower()
    if not text:
        return "empty"

    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "empty"
    if text[0].isdigit():
        text = f"v_{text}"
    return text[:max_length]


def to_serializable(value: Any) -> Any:
    """Convert common Python and pandas objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return parse_datetime(value).isoformat()
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC").isoformat() if value.tzinfo else value.tz_localize("UTC").isoformat()
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return value


def json_dump(payload: Any, destination: Path) -> Path:
    """Persist a JSON payload using UTF-8 encoding."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(to_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


def dataframe_to_csv(dataframe: pd.DataFrame, destination: Path) -> Path:
    """Persist a DataFrame as UTF-8 CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
    return destination
