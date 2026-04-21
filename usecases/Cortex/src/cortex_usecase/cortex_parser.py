"""Parser for persisted Cortex management audit log JSON payloads."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sentinel.ingestion import BaseLogParser

from .cortex_client import extract_records_from_payload, records_to_dataframe


class CortexManagementAuditParser(BaseLogParser):
    """Parse a persisted management audit log payload into a DataFrame."""

    def parse(self) -> pd.DataFrame:
        payload = json.loads(Path(self.file_path).read_text(encoding="utf-8"))
        dataframe = records_to_dataframe(extract_records_from_payload(payload))
        if not dataframe.empty:
            dataframe["__source_file"] = str(Path(self.file_path).resolve())
        return dataframe
