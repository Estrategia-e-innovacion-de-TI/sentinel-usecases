"""Client helpers for Cortex XDR management audit logs."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlsplit

import pandas as pd
import requests

from .utils import parse_datetime, to_epoch_millis


MANAGEMENT_AUDIT_LOGS_PATH = "/public_api/v1/audits/management_logs"
MAX_PAGE_SIZE = 100
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip",
}


class CortexClientError(RuntimeError):
    """Base exception for Cortex use case client errors."""


class CortexConfigurationError(CortexClientError):
    """Raised when the client configuration is incomplete or invalid."""


class CortexHTTPError(CortexClientError):
    """Raised when the Cortex API returns an unexpected HTTP response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class CortexAuthenticationError(CortexHTTPError):
    """Raised when the Cortex API rejects the credentials."""


@dataclass
class CortexClientConfig:
    """Runtime configuration for the Cortex management audit log client."""

    base_url: str
    api_key_id: str
    api_key: str
    auth_mode: str = "advanced"
    verify_ssl: bool = True
    timeout_seconds: int = 30
    page_size: int = MAX_PAGE_SIZE
    request_interval_seconds: float = 0.12

    def __post_init__(self) -> None:
        self.base_url = self.base_url.strip().rstrip("/")
        self.api_key_id = str(self.api_key_id).strip()
        self.api_key = str(self.api_key).strip()
        self.auth_mode = str(self.auth_mode).strip().lower()

        if not self.base_url.startswith(("https://", "http://")):
            raise CortexConfigurationError("XDR_BASE_URL must include the URL scheme, e.g. https://api-{fqdn}.")

        parsed_base_url = urlsplit(self.base_url)
        if not parsed_base_url.netloc:
            raise CortexConfigurationError("XDR_BASE_URL must include a valid host, e.g. https://api-{fqdn}.")

        # Keep only the API origin so users can paste vendor URLs that include an endpoint path.
        self.base_url = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"

        if not self.api_key_id:
            raise CortexConfigurationError("XDR_API_KEY_ID is required.")
        if not self.api_key:
            raise CortexConfigurationError("XDR_API_KEY is required.")
        if self.auth_mode not in {"advanced", "standard"}:
            raise CortexConfigurationError("XDR_AUTH_MODE must be either 'advanced' or 'standard'.")
        if not 1 <= int(self.page_size) <= MAX_PAGE_SIZE:
            raise CortexConfigurationError(f"page_size must be between 1 and {MAX_PAGE_SIZE}.")

    @classmethod
    def from_env(cls) -> "CortexClientConfig":
        """Build the configuration from environment variables."""
        missing = [
            name for name in ("XDR_BASE_URL", "XDR_API_KEY_ID", "XDR_API_KEY")
            if not os.getenv(name)
        ]
        if missing:
            raise CortexConfigurationError(
                "Missing Cortex credentials in the environment: " + ", ".join(sorted(missing))
            )

        verify_ssl = os.getenv("XDR_VERIFY_SSL", "true").strip().lower() not in {"0", "false", "no"}
        timeout_seconds = int(os.getenv("XDR_TIMEOUT_SECONDS", "30"))
        page_size = int(os.getenv("XDR_PAGE_SIZE", str(MAX_PAGE_SIZE)))
        request_interval_seconds = float(os.getenv("XDR_REQUEST_INTERVAL_SECONDS", "0.12"))

        return cls(
            base_url=os.environ["XDR_BASE_URL"],
            api_key_id=os.environ["XDR_API_KEY_ID"],
            api_key=os.environ["XDR_API_KEY"],
            auth_mode=os.getenv("XDR_AUTH_MODE", "advanced"),
            verify_ssl=verify_ssl,
            timeout_seconds=timeout_seconds,
            page_size=page_size,
            request_interval_seconds=request_interval_seconds,
        )


def build_authenticated_headers(config: CortexClientConfig) -> Dict[str, str]:
    """Generate authenticated headers for standard or advanced Cortex API keys."""
    headers = dict(DEFAULT_HEADERS)
    headers["x-xdr-auth-id"] = config.api_key_id

    if config.auth_mode == "standard":
        headers["Authorization"] = config.api_key
        return headers

    nonce = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))
    timestamp = int(time.time() * 1000)
    auth_material = f"{config.api_key}{nonce}{timestamp}".encode("utf-8")
    api_key_hash = hashlib.sha256(auth_material).hexdigest()

    headers["x-xdr-timestamp"] = str(timestamp)
    headers["x-xdr-nonce"] = nonce
    headers["Authorization"] = api_key_hash
    return headers


def build_management_audit_logs_payload(
    *,
    start_time: Optional[Any] = None,
    end_time: Optional[Any] = None,
    offset: int = 0,
    page_size: int = MAX_PAGE_SIZE,
    additional_filters: Optional[Sequence[Dict[str, Any]]] = None,
    sort: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Construct the request payload for the management audit log endpoint."""
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}.")

    request_data: Dict[str, Any] = {
        "search_from": int(offset),
        "search_to": int(offset) + int(page_size),
    }

    filters = []
    if start_time is not None:
        filters.append({"field": "timestamp", "operator": "gte", "value": to_epoch_millis(start_time)})
    if end_time is not None:
        filters.append({"field": "timestamp", "operator": "lte", "value": to_epoch_millis(end_time)})
    if additional_filters:
        filters.extend(additional_filters)
    if filters:
        request_data["filters"] = filters
    if sort:
        request_data["sort"] = sort

    return {"request_data": request_data}


def extract_reply_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``reply`` block if present, otherwise the payload itself."""
    if not isinstance(payload, dict):
        raise TypeError("The Cortex API payload must be a dictionary.")
    reply = payload.get("reply")
    return reply if isinstance(reply, dict) else payload


def extract_records_from_payload(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Extract audit log records from a single response or from a persisted multi-page payload."""
    if not isinstance(payload, dict):
        raise TypeError("The Cortex API payload must be a dictionary.")

    if "records" in payload and isinstance(payload["records"], list):
        return list(payload["records"])

    if "pages" in payload and isinstance(payload["pages"], list):
        records: list[Dict[str, Any]] = []
        for page in payload["pages"]:
            records.extend(extract_records_from_payload(page))
        return records

    reply = extract_reply_block(payload)
    data = reply.get("data", [])
    return list(data) if isinstance(data, list) else []


def _maybe_parse_json_string(value: Any) -> Any:
    """Best-effort decoding for JSON embedded as a string field."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or text[0] not in "[{":
        return value

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def records_to_dataframe(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Normalize raw audit log records into a tabular DataFrame."""
    records = list(records)
    if not records:
        return pd.DataFrame()

    normalized_records: list[Dict[str, Any]] = []
    for record in records:
        normalized_record = dict(record)
        for key, value in list(record.items()):
            parsed_value = _maybe_parse_json_string(value)
            if parsed_value is not value:
                normalized_record[f"{key}_PARSED"] = parsed_value
        normalized_records.append(normalized_record)

    return pd.json_normalize(normalized_records, sep=".")


class CortexManagementAuditClient:
    """Small client focused on Cortex XDR management audit logs."""

    def __init__(
        self,
        config: CortexClientConfig,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()

    @property
    def endpoint_url(self) -> str:
        """Return the fully qualified management audit log endpoint URL."""
        return f"{self.config.base_url}{MANAGEMENT_AUDIT_LOGS_PATH}"

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = build_authenticated_headers(self.config)
        try:
            response = self.session.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds,
                verify=self.config.verify_ssl,
            )
        except requests.RequestException as exc:
            raise CortexHTTPError(
                f"Request to {self.endpoint_url} failed before an HTTP response was received: {exc}"
            ) from exc

        if response.status_code in {401, 403}:
            raise CortexAuthenticationError(
                f"Authentication failed against {self.endpoint_url} (HTTP {response.status_code}). "
                "Validate XDR_BASE_URL, XDR_API_KEY_ID, XDR_API_KEY, and XDR_AUTH_MODE.",
                status_code=response.status_code,
                response_text=response.text[:1000],
            )

        if response.status_code >= 400:
            raise CortexHTTPError(
                f"Cortex API request failed with HTTP {response.status_code} against {self.endpoint_url}.",
                status_code=response.status_code,
                response_text=response.text[:1000],
            )

        try:
            return response.json()
        except ValueError as exc:
            raise CortexHTTPError(
                f"Cortex API returned a non-JSON response from {self.endpoint_url}.",
                status_code=response.status_code,
                response_text=response.text[:1000],
            ) from exc

    def fetch_management_audit_logs_page(
        self,
        *,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        offset: int = 0,
        page_size: Optional[int] = None,
        additional_filters: Optional[Sequence[Dict[str, Any]]] = None,
        sort: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Fetch a single page from the management audit log endpoint."""
        effective_page_size = int(page_size or self.config.page_size)
        payload = build_management_audit_logs_payload(
            start_time=start_time,
            end_time=end_time,
            offset=offset,
            page_size=effective_page_size,
            additional_filters=additional_filters,
            sort=sort,
        )
        return self._request(payload)

    def validate_authentication(
        self,
        *,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Validate credentials by issuing a constrained request to the target endpoint."""
        response = self.fetch_management_audit_logs_page(
            start_time=start_time,
            end_time=end_time,
            offset=0,
            page_size=1,
        )
        reply = extract_reply_block(response)
        return {
            "success": True,
            "total_count": reply.get("total_count"),
            "result_count": reply.get("result_count"),
            "reply_keys": sorted(reply.keys()),
        }

    def fetch_management_audit_logs(
        self,
        *,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        initial_offset: int = 0,
        page_size: Optional[int] = None,
        max_records: int = MAX_PAGE_SIZE,
        additional_filters: Optional[Sequence[Dict[str, Any]]] = None,
        sort: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Paginate over management audit logs and return both metadata and raw pages."""
        effective_page_size = min(int(page_size or self.config.page_size), MAX_PAGE_SIZE)
        records: list[Dict[str, Any]] = []
        pages: list[Dict[str, Any]] = []
        page_summaries: list[Dict[str, Any]] = []
        offset = int(initial_offset)
        started_at = time.time()
        resolved_start = parse_datetime(start_time)
        resolved_end = parse_datetime(end_time)

        while len(records) < int(max_records):
            remaining = int(max_records) - len(records)
            current_page_size = min(effective_page_size, remaining)
            response = self.fetch_management_audit_logs_page(
                start_time=resolved_start,
                end_time=resolved_end,
                offset=offset,
                page_size=current_page_size,
                additional_filters=additional_filters,
                sort=sort,
            )
            reply = extract_reply_block(response)
            page_records = extract_records_from_payload(response)

            pages.append(response)
            records.extend(page_records)
            page_summaries.append(
                {
                    "offset": offset,
                    "requested_page_size": current_page_size,
                    "returned_records": len(page_records),
                    "total_count": reply.get("total_count"),
                }
            )

            if len(page_records) < current_page_size:
                break

            total_count = reply.get("total_count")
            offset += current_page_size
            if total_count is not None and offset >= int(total_count):
                break

            time.sleep(self.config.request_interval_seconds)

        return {
            "metadata": {
                "endpoint_url": self.endpoint_url,
                "start_time": resolved_start.isoformat() if resolved_start else None,
                "end_time": resolved_end.isoformat() if resolved_end else None,
                "initial_offset": initial_offset,
                "page_size": effective_page_size,
                "max_records": int(max_records),
                "record_count": len(records),
                "page_count": len(pages),
                "elapsed_seconds": round(time.time() - started_at, 3),
            },
            "page_summaries": page_summaries,
            "pages": pages,
            "records": records,
        }
