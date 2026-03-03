"""Ingestion module for parsing raw log files into structured DataFrames.

Provides a base parser class and specific parsers for common
enterprise log formats:

- WAS (WebSphere Application Server)
- HSM (Hardware Security Module)
- HDC (High-Density Computing)
- IBMMQ (IBM Message Queue)
- ZTNA (Cloudflare Zero Trust Network Access)

Custom parsers can be created by extending ``BaseLogParser``.
"""

from .ingestor import LogIngestor
from .base_parser import BaseLogParser
from .hdc_parser import HDCParser
from .hsm_parser import HSMParser
from .ibmmq_parser import IBMMQParser
from .was_parser import WASParser
from .ztna_parser import ZTNAParser

__all__ = [
    "LogIngestor",
    "BaseLogParser",
    "HDCParser",
    "HSMParser",
    "IBMMQParser",
    "WASParser",
    "ZTNAParser",
]
