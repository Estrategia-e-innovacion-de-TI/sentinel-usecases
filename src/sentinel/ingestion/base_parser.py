"""Abstract base class for log parsers."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseLogParser(ABC):
    """Abstract base class defining the interface for log parsers.

    Subclasses must implement the ``parse`` method to convert a raw
    log file into a structured ``pd.DataFrame``.

    Parameters
    ----------
    file_path : str
        Path to the log file to parse.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """Parse the log file and return structured data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the parsed log entries.
        """
        pass
