"""Convenience class for dispatching log parsing by type."""

from .hdc_parser import HDCParser
from .hsm_parser import HSMParser
from .ibmmq_parser import IBMMQParser
from .was_parser import WASParser
from .ztna_parser import ZTNAParser


class LogIngestor:
    """Dispatch log file parsing based on log type identifier.

    Attributes
    ----------
    parsers : dict
        Mapping of log type strings to their parser classes.
    """

    parsers = {
        "HSM": HSMParser,
        "HDC": HDCParser,
        "IBM_MQ": IBMMQParser,
        "WAS": WASParser,
        "ZTNA": ZTNAParser,
    }

    @classmethod
    def ingest(cls, file_path: str, log_type: str):
        """Parse a log file using the appropriate parser.

        Parameters
        ----------
        file_path : str
            Path to the log file.
        log_type : str
            Log type identifier (e.g. ``'HSM'``, ``'HDC'``, ``'IBM_MQ'``,
            ``'WAS'``, ``'ZTNA'``). Case-insensitive.

        Returns
        -------
        pd.DataFrame
            Parsed log data.

        Raises
        ------
        ValueError
            If ``log_type`` is not supported.
        """
        parser_class = cls.parsers.get(log_type.upper())
        if not parser_class:
            raise ValueError(f"Unsupported log type: {log_type}")
        parser = parser_class(file_path)
        return parser.parse()
