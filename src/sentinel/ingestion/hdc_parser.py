import re
import pandas as pd
from .base_parser import BaseLogParser

LOG_PATTERN = re.compile(
    r"\[(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}:\d{3}) COT\]\s+(\w+)\s+(\w+)\s+(\w)\s+(.+)",
    re.DOTALL
)

def extract_error_code(description):
    """Extract an error code pattern from a log description string.

    Parameters
    ----------
    description : str
        Log message text.

    Returns
    -------
    str or None
        Matched error code, or ``None`` if not found.
    """
    match = re.search(r'([A-Z]+\d+[A-Z]+:)', description)
    return match.group(1) if match else None


class HDCParser(BaseLogParser):
    """Parser for High-Density Computing (HDC) log files."""

    def parse(self) -> pd.DataFrame:
        """Parse the HDC log file into a structured DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``thread_id``, ``log_source``,
            ``log_type``, ``message``, ``error_code``.
            Returns an empty DataFrame if the file cannot be read.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                raw_log = file.read()
        except Exception:
            return pd.DataFrame()

        log_lines = [line.strip() for line in raw_log.strip().split('\n') if line.strip()]
        data = []
        for line in log_lines:
            match = LOG_PATTERN.match(line)
            if match:
                data.append({
                    'timestamp': match.group(1),
                    'thread_id': match.group(2),
                    'log_source': match.group(3),
                    'log_type': match.group(4),
                    'message': match.group(5),
                    'error_code': extract_error_code(match.group(5))
                })
        return pd.DataFrame(data)