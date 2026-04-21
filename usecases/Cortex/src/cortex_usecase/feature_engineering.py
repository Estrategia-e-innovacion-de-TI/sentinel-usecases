"""Schema analysis, feature engineering, and Sentinel integration helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from sentinel.detectors import IsolationForestDetector, RRCFDetector
from sentinel.explorer import SignalDiagnostics, Thresholds
from sentinel.transformer import RollingAggregator, StringAggregator

from .utils import sanitize_category_name


FIELD_ALIAS_CANDIDATES: Dict[str, Sequence[str]] = {
    "event_time": ("AUDIT_INSERT_TIME", "timestamp", "event_time", "end"),
    "owner_name": ("AUDIT_OWNER_NAME", "owner_name", "user", "suser"),
    "owner_email": ("AUDIT_OWNER_EMAIL", "owner_email", "email", "user_email", "cs1"),
    "entity": ("AUDIT_ENTITY", "entity", "type", "name"),
    "entity_subtype": ("AUDIT_ENTITY_SUBTYPE", "entity_subtype", "sub_type", "subtype", "cs2"),
    "result": ("AUDIT_RESULT", "result", "outcome", "cs3"),
    "reason": ("AUDIT_REASON", "reason", "failure_reason", "cs4"),
    "description": ("AUDIT_DESCRIPTION", "description", "msg", "message"),
    "severity": ("AUDIT_SEVERITY", "severity"),
    "hostname": ("AUDIT_HOSTNAME", "hostname", "host"),
    "asset_names": ("AUDIT_ASSET_NAMES", "asset_names"),
    "asset_json": ("AUDIT_ASSET_JSON", "asset_json"),
    "session_id": ("AUDIT_SESSION_ID", "session_id"),
    "case_id": ("AUDIT_CASE_ID", "case_id"),
}


RAW_FIELD_HINTS: Dict[str, Dict[str, str]] = {
    "AUDIT_ID": {
        "descripcion_inferida": "Identificador único del evento administrativo.",
        "relevancia_analitica": "Media",
        "uso_potencial": "Trazabilidad, deduplicación y correlación con artefactos exportados.",
        "fortaleza_esperada": "Alta para integridad del dataset.",
        "limitaciones": "No aporta señal de anomalía por sí solo.",
    },
    "AUDIT_OWNER_NAME": {
        "descripcion_inferida": "Nombre visible del actor administrativo.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "Perfiles de comportamiento por administrador y secuencias por usuario.",
        "fortaleza_esperada": "Media a alta según consistencia operativa.",
        "limitaciones": "Puede variar por alias, cambios de display name o cuentas compartidas.",
    },
    "AUDIT_OWNER_EMAIL": {
        "descripcion_inferida": "Correo del actor administrativo.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "Identidad primaria para baselines, rareza y first-seen.",
        "fortaleza_esperada": "Alta si el tenant usa identidades personales.",
        "limitaciones": "Se debilita con service accounts o automatizaciones impersonales.",
    },
    "AUDIT_RESULT": {
        "descripcion_inferida": "Resultado de la acción administrativa.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "Detección de bursts de fallo, pruebas erróneas y cambios rechazados.",
        "fortaleza_esperada": "Alta cuando existe taxonomía consistente de estados.",
        "limitaciones": "Las etiquetas exactas pueden variar entre tipos y versiones.",
    },
    "AUDIT_REASON": {
        "descripcion_inferida": "Motivo o detalle resumido del resultado.",
        "relevancia_analitica": "Media",
        "uso_potencial": "Clustering de fallos, causas recurrentes y semántica de errores.",
        "fortaleza_esperada": "Media.",
        "limitaciones": "Texto libre, con nulos frecuentes y baja estandarización.",
    },
    "AUDIT_DESCRIPTION": {
        "descripcion_inferida": "Descripción textual del cambio o acción ejecutada.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "Banderas semánticas para uninstall, token, API key, config y Action Center.",
        "fortaleza_esperada": "Alta para enriquecer contexto de anomalías.",
        "limitaciones": "Puede ser largo, ruidoso o cambiar entre releases.",
    },
    "AUDIT_ENTITY": {
        "descripcion_inferida": "Familia o tipo principal de evento administrativo.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "Volumetría por dominio, rareza de acciones y detección de picos por categoría.",
        "fortaleza_esperada": "Alta.",
        "limitaciones": "Granularidad media; por sí sola puede ocultar subtipos críticos.",
    },
    "AUDIT_ENTITY_SUBTYPE": {
        "descripcion_inferida": "Subtipo más específico del evento administrativo.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "Rare actions, cambios críticos concretos y secuencias operativas.",
        "fortaleza_esperada": "Alta cuando la cardinalidad es manejable.",
        "limitaciones": "Puede ser muy disperso y requerir normalización semántica.",
    },
    "AUDIT_SESSION_ID": {
        "descripcion_inferida": "Identificador de sesión asociado a la actividad.",
        "relevancia_analitica": "Media",
        "uso_potencial": "Secuencias, agrupación por sesión y pivot con múltiples acciones rápidas.",
        "fortaleza_esperada": "Media.",
        "limitaciones": "No siempre estará presente ni estable entre tipos de acción.",
    },
    "AUDIT_CASE_ID": {
        "descripcion_inferida": "Caso relacionado, si aplica.",
        "relevancia_analitica": "Baja",
        "uso_potencial": "Correlación con flujos de incidentes o war rooms.",
        "fortaleza_esperada": "Baja a media.",
        "limitaciones": "Suele ser nulo fuera de dominios orientados a casos.",
    },
    "AUDIT_INSERT_TIME": {
        "descripcion_inferida": "Marca temporal del evento administrativo.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "Series temporales, estacionalidad, bursts y secuencias.",
        "fortaleza_esperada": "Alta.",
        "limitaciones": "Hay que confirmar zona horaria y semántica exacta del timestamp.",
    },
    "AUDIT_SEVERITY": {
        "descripcion_inferida": "Severidad asociada al evento.",
        "relevancia_analitica": "Media",
        "uso_potencial": "Priorización y scoring contextual.",
        "fortaleza_esperada": "Media.",
        "limitaciones": "No sustituye el contexto operativo real ni el blast radius.",
    },
    "AUDIT_HOSTNAME": {
        "descripcion_inferida": "Hostname afectado, si existe impacto sobre un endpoint concreto.",
        "relevancia_analitica": "Media",
        "uso_potencial": "Estimación de alcance y detección de acciones concentradas en hosts.",
        "fortaleza_esperada": "Media.",
        "limitaciones": "Puede estar ausente en eventos puramente administrativos.",
    },
    "AUDIT_ASSET_NAMES": {
        "descripcion_inferida": "Activos o nombres de activos asociados a la acción.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "Blast radius, uninstall masivo y cambios con impacto de cobertura.",
        "fortaleza_esperada": "Alta cuando viene poblado.",
        "limitaciones": "Formato y delimitación pueden variar; requiere parsing adicional.",
    },
    "AUDIT_ASSET_JSON": {
        "descripcion_inferida": "Detalle estructurado del activo o assets impactados.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "Correlación fina con endpoints, tags o grupos afectados.",
        "fortaleza_esperada": "Alta si el JSON es consistente.",
        "limitaciones": "Puede venir como string serializado y requerir flattening.",
    },
    "SOURCE_IP": {
        "descripcion_inferida": "IP de origen observada para la acción administrativa.",
        "relevancia_analitica": "Alta",
        "uso_potencial": "First-seen por usuario, rareza de origen y posibles automatizaciones no habituales.",
        "fortaleza_esperada": "Alta si la IP está presente y estable.",
        "limitaciones": "Puede ocultarse detrás de proxys, NAT o planos de control internos.",
    },
    "USER_AGENT": {
        "descripcion_inferida": "User-Agent del cliente que originó la acción.",
        "relevancia_analitica": "Media",
        "uso_potencial": "Tooling novelty, automatización y acceso desde clientes no habituales.",
        "fortaleza_esperada": "Media.",
        "limitaciones": "A menudo es ruidoso, mutable o puede faltar por completo.",
    },
}


SEMANTIC_PATTERNS: Dict[str, str] = {
    "mentions_uninstall": r"\buninstall|\bremove agent|\bdelete endpoint|\bdeactivate agent",
    "mentions_token": r"\btoken\b",
    "mentions_api_key": r"\bapi[\s_-]?key\b",
    "mentions_config": r"\bconfig|\bconfiguration|\bpolicy|\bprofile|\bsetting|\brule",
    "mentions_action_center": r"\baction center\b",
    "mentions_permissions": r"\bpermission|\brole|\bscope|\baccess\b",
    "mentions_authentication": r"\bauth|\blogin|\bsso\b",
}


def _find_alias(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match:
            return match
    return None


def _find_by_tokens(columns: Sequence[str], required_tokens: Sequence[str]) -> Optional[str]:
    for column in columns:
        lowered = column.lower()
        if all(token in lowered for token in required_tokens):
            return column
    return None


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.to_datetime(series, utc=True)

    numeric = pd.to_numeric(series, errors="coerce")
    parsed_numeric = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
    parsed_text = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed_numeric.fillna(parsed_text)


def _combine_text_fields(dataframe: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    parts = [dataframe[column].fillna("").astype(str) for column in columns]
    return parts[0].str.cat(parts[1:], sep=" | ").str.strip(" |")


def _stability_guess(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "No observable con el dataset actual."

    cardinality = _hashable_series(non_null).nunique(dropna=True)
    uniqueness_ratio = cardinality / max(len(non_null), 1)
    if pd.api.types.is_numeric_dtype(non_null):
        return "Variable continua o de alta granularidad."
    if uniqueness_ratio <= 0.05:
        return "Estable; buena candidata para baselines categóricos."
    if uniqueness_ratio <= 0.5:
        return "Moderadamente variable; útil con agregación por ventana."
    return "Alta cardinalidad; conviene normalización o agrupación."


def _field_hint(column_name: str) -> Dict[str, str]:
    hint = RAW_FIELD_HINTS.get(column_name)
    if hint:
        return hint

    lowered = column_name.lower()
    if "ip" in lowered:
        return RAW_FIELD_HINTS["SOURCE_IP"]
    if "agent" in lowered and "user" in lowered:
        return RAW_FIELD_HINTS["USER_AGENT"]
    if "time" in lowered or "timestamp" in lowered:
        return RAW_FIELD_HINTS["AUDIT_INSERT_TIME"]
    if "result" in lowered or "status" in lowered:
        return RAW_FIELD_HINTS["AUDIT_RESULT"]
    if "description" in lowered or "message" in lowered:
        return RAW_FIELD_HINTS["AUDIT_DESCRIPTION"]
    if "type" in lowered and "sub" in lowered:
        return RAW_FIELD_HINTS["AUDIT_ENTITY_SUBTYPE"]
    if "type" in lowered or "entity" in lowered:
        return RAW_FIELD_HINTS["AUDIT_ENTITY"]
    if "owner" in lowered or "email" in lowered or "user" in lowered:
        return RAW_FIELD_HINTS["AUDIT_OWNER_EMAIL"]

    return {
        "descripcion_inferida": "Campo no documentado explícitamente; requiere validación con datos reales.",
        "relevancia_analitica": "Media",
        "uso_potencial": "Exploración de esquema, correlación y enriquecimiento contextual.",
        "fortaleza_esperada": "Por validar.",
        "limitaciones": "Interpretación heurística hasta ver suficiente histórico.",
    }


def normalize_management_audit_dataframe(
    dataframe: pd.DataFrame,
    *,
    local_timezone: str = "America/Bogota",
    business_hours_start: int = 7,
    business_hours_end: int = 19,
    high_risk_sequence_window_minutes: int = 30,
) -> pd.DataFrame:
    """Map raw Cortex audit fields into a canonical event model."""
    if dataframe.empty:
        return pd.DataFrame()

    working = dataframe.copy()
    available_columns = list(working.columns)
    selected_columns = {
        name: _find_alias(available_columns, candidates)
        for name, candidates in FIELD_ALIAS_CANDIDATES.items()
    }

    selected_columns["source_ip"] = (
        _find_alias(available_columns, ("SOURCE_IP", "source_ip", "ip_address"))
        or _find_by_tokens(available_columns, ("source", "ip"))
        or _find_by_tokens(available_columns, ("client", "ip"))
    )
    selected_columns["user_agent"] = (
        _find_alias(available_columns, ("USER_AGENT", "user_agent", "useragent"))
        or _find_by_tokens(available_columns, ("user", "agent"))
        or _find_by_tokens(available_columns, ("client", "agent"))
    )

    for canonical_name, source_column in selected_columns.items():
        working[canonical_name] = working[source_column] if source_column else pd.NA

    working["event_time"] = _parse_timestamp_series(working["event_time"])
    working = working.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)
    if working.empty:
        return working

    working["event_time_local"] = working["event_time"].dt.tz_convert(local_timezone)
    working["owner_identity"] = (
        working["owner_email"].fillna(working["owner_name"]).fillna("unknown_owner").astype(str)
    )
    working["entity"] = working["entity"].fillna("unknown_entity").astype(str)
    working["entity_subtype"] = working["entity_subtype"].fillna("unknown_subtype").astype(str)
    working["result"] = working["result"].fillna("UNKNOWN").astype(str)
    working["reason"] = working["reason"].fillna("").astype(str)
    working["description"] = working["description"].fillna("").astype(str)
    working["description_text"] = _combine_text_fields(
        working,
        ("entity", "entity_subtype", "description", "reason"),
    )
    working["semantic_text"] = working["description_text"].str.lower()
    working["result_normalized"] = working["result"].str.upper().str.strip()

    failure_pattern = r"FAIL|ERROR|DENY|TIMEOUT|UNAUTHORIZED|FORBIDDEN|INVALID"
    working["is_failure"] = working["result_normalized"].str.contains(failure_pattern, regex=True, na=False)
    working["is_success"] = working["result_normalized"].str.contains(r"SUCCESS|OK|DONE|COMPLETED", regex=True, na=False)

    for column_name, pattern in SEMANTIC_PATTERNS.items():
        working[column_name] = working["semantic_text"].str.contains(pattern, regex=True, na=False)

    working["is_high_risk_action"] = (
        working["mentions_uninstall"]
        | working["mentions_token"]
        | working["mentions_api_key"]
        | working["mentions_config"]
        | working["mentions_action_center"]
        | working["mentions_permissions"]
    )

    working["event_hour_local"] = working["event_time_local"].dt.hour
    working["event_dayofweek_local"] = working["event_time_local"].dt.day_name()
    working["is_business_day"] = working["event_time_local"].dt.dayofweek < 5
    working["is_outside_business_hours"] = (
        ~working["is_business_day"]
        | (working["event_hour_local"] < business_hours_start)
        | (working["event_hour_local"] >= business_hours_end)
    )

    working["entity_subtype_key"] = working["entity"] + " :: " + working["entity_subtype"]
    action_frequencies = working["entity_subtype_key"].value_counts(dropna=False)
    rare_cutoff = max(2, min(5, int(max(1, len(working) * 0.03))))
    working["is_rare_action"] = working["entity_subtype_key"].map(action_frequencies).fillna(0).le(rare_cutoff)

    for suffix in ("source_ip", "user_agent"):
        if working[suffix].notna().any():
            valid_rows = working["owner_identity"].notna() & working[suffix].notna()
            duplicated_pairs = working.loc[valid_rows].duplicated(subset=["owner_identity", suffix], keep="first")
            flag = pd.Series(False, index=working.index)
            flag.loc[valid_rows] = ~duplicated_pairs
            working[f"is_first_seen_{suffix}_for_user"] = flag
        else:
            working[f"is_first_seen_{suffix}_for_user"] = False

    grouped = working.groupby("owner_identity", dropna=False)
    working["previous_event_time"] = grouped["event_time"].shift(1)
    working["previous_entity"] = grouped["entity"].shift(1).fillna("")
    working["previous_result_normalized"] = grouped["result_normalized"].shift(1).fillna("")
    working["previous_is_failure"] = grouped["is_failure"].shift(1, fill_value=False)
    working["previous_is_high_risk_action"] = grouped["is_high_risk_action"].shift(1, fill_value=False)
    working["previous_mentions_token"] = grouped["mentions_token"].shift(1, fill_value=False)
    working["previous_mentions_api_key"] = grouped["mentions_api_key"].shift(1, fill_value=False)
    working["previous_mentions_permissions"] = grouped["mentions_permissions"].shift(1, fill_value=False)
    working["previous_mentions_config"] = grouped["mentions_config"].shift(1, fill_value=False)

    window_seconds = high_risk_sequence_window_minutes * 60
    working["seconds_since_previous_user_event"] = (
        working["event_time"] - working["previous_event_time"]
    ).dt.total_seconds()
    rapid_sequence = working["seconds_since_previous_user_event"].le(window_seconds).fillna(False)
    working["is_high_risk_sequence"] = rapid_sequence & (
        (working["previous_is_failure"] & working["is_success"])
        | (working["previous_is_high_risk_action"] & working["is_high_risk_action"])
        | (working["previous_mentions_token"] & (working["mentions_api_key"] | working["mentions_config"] | working["mentions_uninstall"]))
        | (working["previous_mentions_api_key"] & (working["mentions_config"] | working["mentions_uninstall"]))
        | (working["previous_mentions_permissions"] & (working["mentions_api_key"] | working["mentions_config"] | working["mentions_uninstall"]))
        | (working["previous_mentions_config"] & working["mentions_uninstall"])
    )
    working["sequence_signature"] = (
        working["previous_entity"].astype(str).str.strip()
        + " -> "
        + working["entity"].astype(str).str.strip()
    )

    return working


def build_schema_profile(dataframe: pd.DataFrame, max_examples: int = 3) -> pd.DataFrame:
    """Summarize columns, types, nulls, cardinality, stability, and analytic utility."""
    rows = []
    for column in dataframe.columns:
        series = dataframe[column]
        non_null = int(series.notna().sum())
        cardinality = int(_hashable_series(series).dropna().nunique()) if non_null else 0
        examples = [str(value)[:120] for value in series.dropna().astype(str).head(max_examples).tolist()]
        hint = _field_hint(column)
        rows.append(
            {
                "field": column,
                "dtype": str(series.dtype),
                "non_null_count": non_null,
                "null_pct": round(float(series.isna().mean() * 100), 2),
                "cardinality": cardinality,
                "sample_values": " | ".join(examples),
                "stability_guess": _stability_guess(series),
                "analytic_utility": hint["uso_potencial"],
            }
        )

    return pd.DataFrame(rows).sort_values(["null_pct", "cardinality", "field"]).reset_index(drop=True)


def _make_hashable(value: Any) -> Any:
    """Normalize nested values so they can be counted safely by pandas."""
    if isinstance(value, list):
        return tuple(_make_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _make_hashable(item)) for key, item in value.items()))
    if isinstance(value, set):
        return tuple(sorted(_make_hashable(item) for item in value))
    return value


def _hashable_series(series: pd.Series) -> pd.Series:
    """Convert non-scalar values into hashable equivalents for uniqueness checks."""
    return series.map(_make_hashable)


def build_field_interpretation_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the required field interpretation table for the notebook."""
    rows = []
    for column in dataframe.columns:
        hint = _field_hint(column)
        rows.append({"campo": column, **hint})

    relevance_order = {"Alta": 0, "Media": 1, "Baja": 2}
    table = pd.DataFrame(rows)
    table["_order"] = table["relevancia_analitica"].map(relevance_order).fillna(3)
    table = table.sort_values(["_order", "campo"]).drop(columns="_order").reset_index(drop=True)
    return table


def _category_counts(
    dataframe: pd.DataFrame,
    *,
    column: str,
    time_window: str,
    top_n: int,
    prefix: str,
) -> pd.DataFrame:
    if column not in dataframe.columns:
        return pd.DataFrame(index=pd.Index([], name="window_start"))

    populated = dataframe[column].dropna().astype(str)
    if populated.empty:
        return pd.DataFrame(index=pd.Index([], name="window_start"))

    top_categories = populated.value_counts().head(top_n).index.tolist()
    counts = (
        dataframe.loc[dataframe[column].isin(top_categories)]
        .groupby([pd.Grouper(key="event_time", freq=time_window), column])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(columns=top_categories, fill_value=0)
    counts.index.name = "window_start"
    counts.columns = [f"{prefix}_count__{sanitize_category_name(value)}" for value in counts.columns]
    return counts


def build_window_features(
    dataframe: pd.DataFrame,
    *,
    time_window: str = "15min",
    rolling_window_size: int = 4,
    top_entities: int = 5,
    top_subtypes: int = 5,
    top_users: int = 3,
) -> pd.DataFrame:
    """Aggregate normalized audit events into time-windowed Sentinel features."""
    if dataframe.empty:
        return pd.DataFrame()

    working = dataframe.copy()
    if "event_time" not in working.columns:
        working = normalize_management_audit_dataframe(working)
    if working.empty:
        return pd.DataFrame()

    aggregator = StringAggregator(working, timestamp_column="event_time")
    column_metrics = {
        "owner_identity": ["count", "nunique"],
        "entity": ["nunique"],
        "entity_subtype_key": ["nunique"],
    }
    for optional_column in ("source_ip", "user_agent", "hostname"):
        if optional_column in working.columns and working[optional_column].notna().any():
            column_metrics[optional_column] = ["nunique"]

    custom_metrics = {
        "failure_count": lambda group: int(group["is_failure"].sum()),
        "outside_business_hours_count": lambda group: int(group["is_outside_business_hours"].sum()),
        "first_seen_ip_count": lambda group: int(group["is_first_seen_source_ip_for_user"].sum()),
        "first_seen_user_agent_count": lambda group: int(group["is_first_seen_user_agent_for_user"].sum()),
        "rare_action_count": lambda group: int(group["is_rare_action"].sum()),
        "high_risk_action_count": lambda group: int(group["is_high_risk_action"].sum()),
        "high_risk_sequence_count": lambda group: int(group["is_high_risk_sequence"].sum()),
        "uninstall_flag_count": lambda group: int(group["mentions_uninstall"].sum()),
        "token_flag_count": lambda group: int(group["mentions_token"].sum()),
        "api_key_flag_count": lambda group: int(group["mentions_api_key"].sum()),
        "config_flag_count": lambda group: int(group["mentions_config"].sum()),
        "action_center_flag_count": lambda group: int(group["mentions_action_center"].sum()),
        "permissions_flag_count": lambda group: int(group["mentions_permissions"].sum()),
        "authentication_flag_count": lambda group: int(group["mentions_authentication"].sum()),
        "failed_admins_nunique": lambda group: int(group.loc[group["is_failure"], "owner_identity"].nunique()),
    }

    if "asset_names" in working.columns and working["asset_names"].notna().any():
        custom_metrics["assets_nunique"] = lambda group: int(group["asset_names"].astype(str).nunique())

    aggregated = aggregator.create_time_aggregation(
        time_window=time_window,
        column_metrics=column_metrics,
        custom_metrics=custom_metrics,
    )
    aggregated.index.name = "window_start"
    aggregated = aggregated.rename(
        columns={
            "owner_identity_count": "events_count",
            "owner_identity_nunique": "unique_admins",
            "entity_nunique": "unique_entities",
            "entity_subtype_key_nunique": "unique_entity_subtypes",
            "source_ip_nunique": "unique_ips",
            "user_agent_nunique": "unique_user_agents",
            "hostname_nunique": "unique_hostnames",
        }
    )

    top_frames = [
        _category_counts(working, column="entity", time_window=time_window, top_n=top_entities, prefix="entity"),
        _category_counts(
            working,
            column="entity_subtype_key",
            time_window=time_window,
            top_n=top_subtypes,
            prefix="subtype",
        ),
        _category_counts(
            working,
            column="owner_identity",
            time_window=time_window,
            top_n=top_users,
            prefix="user",
        ),
    ]

    features = aggregated.copy()
    for frame in top_frames:
        if not frame.empty:
            features = features.join(frame, how="left")

    count_columns = [column for column in features.columns if column.endswith("_count")]
    features[count_columns] = features[count_columns].fillna(0)
    features = features.fillna(0)

    features["failure_ratio"] = np.where(features["events_count"] > 0, features["failure_count"] / features["events_count"], 0.0)
    features["outside_business_hours_ratio"] = np.where(
        features["events_count"] > 0,
        features["outside_business_hours_count"] / features["events_count"],
        0.0,
    )
    features["high_risk_action_ratio"] = np.where(
        features["events_count"] > 0,
        features["high_risk_action_count"] / features["events_count"],
        0.0,
    )
    features["burstiness_index"] = (
        features["events_count"] / features["avg_time_between_events_seconds"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    rolling_targets = [
        column for column in (
            "events_count",
            "failure_count",
            "unique_admins",
            "unique_ips",
            "high_risk_action_count",
            "uninstall_flag_count",
            "token_flag_count",
            "api_key_flag_count",
        )
        if column in features.columns
    ]
    if rolling_targets:
        rolling = RollingAggregator(
            window_size=max(1, int(rolling_window_size)),
            aggregation_functions=["mean", "std"],
            columns=rolling_targets,
            min_periods=1,
        ).fit_transform(features[rolling_targets])
        derived_columns = [column for column in rolling.columns if column not in rolling_targets]
        features = pd.concat([features, rolling[derived_columns]], axis=1)

        for column in rolling_targets:
            mean_column = f"{column}_rolling_mean"
            std_column = f"{column}_rolling_std"
            if mean_column in features.columns:
                features[f"{column}_delta_from_rolling_mean"] = features[column] - features[mean_column]
            if mean_column in features.columns and std_column in features.columns:
                denominator = features[std_column].replace(0, np.nan)
                features[f"{column}_zscore_like"] = (
                    (features[column] - features[mean_column]) / denominator
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return features.sort_index()


def prepare_detection_matrix(
    feature_dataframe: pd.DataFrame,
    *,
    exclude_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Select numeric columns suitable for model-based anomaly detection."""
    exclude = set(exclude_columns or ())
    numeric = feature_dataframe.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=[column for column in exclude if column in numeric.columns], errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    informative_columns = [column for column in numeric.columns if numeric[column].nunique(dropna=True) > 1]
    return numeric[informative_columns]


def run_signal_review(
    feature_dataframe: pd.DataFrame,
    *,
    thresholds: Optional[Thresholds] = None,
) -> Dict[str, Any]:
    """Run Sentinel Explorer on the engineered features."""
    matrix = prepare_detection_matrix(feature_dataframe)
    if matrix.empty:
        raise ValueError("No numeric and non-constant features are available for SignalDiagnostics.")

    diagnostics = SignalDiagnostics(matrix.reset_index(drop=True), columns=list(matrix.columns))
    report = diagnostics.quality_report(thresholds or Thresholds.relaxed())
    summary = pd.DataFrame(diagnostics.summary()).T.reset_index().rename(columns={"index": "feature"})

    return {
        "matrix": matrix,
        "diagnostics": diagnostics,
        "summary": summary,
        "quality_report": report,
        "interpretation": report.interpret(),
    }


def run_isolation_forest_detection(
    feature_dataframe: pd.DataFrame,
    *,
    contamination: float = 0.08,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run Sentinel's IsolationForest detector on the engineered features."""
    matrix = prepare_detection_matrix(feature_dataframe)
    if len(matrix) < 8:
        raise ValueError("Isolation Forest requires at least 8 windows for a meaningful dry run.")

    detector = IsolationForestDetector(
        contamination=contamination,
        random_state=random_state,
    )
    detector.fit(matrix)

    detected = feature_dataframe.copy()
    detected["iforest_label"] = detector.predict(matrix)
    detected["iforest_score_raw"] = detector.decision_function(matrix)
    detected["iforest_score"] = -detected["iforest_score_raw"]

    anomalous_scores = detected.loc[detected["iforest_label"] == -1, "iforest_score"]
    threshold = float(anomalous_scores.min()) if not anomalous_scores.empty else float(detected["iforest_score"].quantile(0.95))

    return {
        "detected": detected,
        "matrix": matrix,
        "detector": detector,
        "threshold": threshold,
    }


def run_rrcf_detection(
    feature_dataframe: pd.DataFrame,
    *,
    target_column: str = "events_count",
    score_quantile: float = 0.95,
) -> Optional[Dict[str, Any]]:
    """Optionally run Sentinel's RRCF detector if the dependency is installed."""
    if target_column not in feature_dataframe.columns or len(feature_dataframe) < 8:
        return None

    series = feature_dataframe[target_column].fillna(0.0)
    try:
        detector = RRCFDetector()
        scores = detector.fit_predict(series)
    except ImportError:
        return None

    detected = feature_dataframe.copy()
    detected["rrcf_score"] = scores.values if hasattr(scores, "values") else np.asarray(scores)
    threshold = float(detected["rrcf_score"].quantile(score_quantile))
    detected["rrcf_label"] = np.where(detected["rrcf_score"] >= threshold, -1, 1)

    return {
        "detected": detected,
        "detector": detector,
        "threshold": threshold,
        "target_column": target_column,
    }


def link_events_to_anomalous_windows(
    event_dataframe: pd.DataFrame,
    detection_dataframe: pd.DataFrame,
    *,
    time_window: str = "15min",
    label_column: str = "iforest_label",
    score_column: str = "iforest_score",
) -> pd.DataFrame:
    """Attach detected anomaly windows back to the original event-level records."""
    if event_dataframe.empty or detection_dataframe.empty or label_column not in detection_dataframe.columns:
        return pd.DataFrame()

    anomalous_windows = detection_dataframe.loc[detection_dataframe[label_column] == -1].copy()
    if anomalous_windows.empty:
        return pd.DataFrame()

    mapping = anomalous_windows[[score_column]].copy()
    mapping.index.name = "window_start"
    mapping = mapping.reset_index()

    events = event_dataframe.copy()
    if "event_time" not in events.columns:
        events = normalize_management_audit_dataframe(events)
    if events.empty:
        return pd.DataFrame()

    events["window_start"] = events["event_time"].dt.floor(time_window)
    return events.merge(mapping, on="window_start", how="inner").sort_values(["window_start", "event_time"])


def build_anomaly_taxonomy_table(
    event_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create the required anomaly taxonomy table for the notebook."""
    if event_dataframe.empty:
        return pd.DataFrame(
            columns=[
                "tipo_de_anomalia",
                "detectabilidad_con_dataset_actual",
                "features_candidatas",
                "dependencia_de_mayor_historico",
                "prioridad_siguiente_iteracion",
            ]
        )

    normalized = event_dataframe if "event_time" in event_dataframe.columns else normalize_management_audit_dataframe(event_dataframe)
    has_ip = "source_ip" in normalized.columns and normalized["source_ip"].notna().any()
    has_user_agent = "user_agent" in normalized.columns and normalized["user_agent"].notna().any()
    history_hours = (
        (normalized["event_time"].max() - normalized["event_time"].min()).total_seconds() / 3600
        if len(normalized) > 1
        else 0.0
    )

    def detectability(*, requires_ip: bool = False, requires_user_agent: bool = False, needs_long_history: bool = False) -> str:
        missing = []
        if requires_ip and not has_ip:
            missing.append("IP")
        if requires_user_agent and not has_user_agent:
            missing.append("user-agent")
        if missing:
            return "Parcial: faltan campos " + " y ".join(missing)
        if needs_long_history and history_hours < 24:
            return "Media en validación rápida; mejora de forma importante con más histórico"
        if history_hours < 6:
            return "Viable solo como dry run técnico"
        if history_hours < 24:
            return "Media"
        return "Alta"

    rows = [
        {
            "tipo_de_anomalia": "Picos de volumen administrativo",
            "detectabilidad_con_dataset_actual": detectability(),
            "features_candidatas": "events_count, entity_count__*, unique_admins, events_count_zscore_like",
            "dependencia_de_mayor_historico": "Deseable para separar bursts legítimos de mantenimiento programado.",
            "prioridad_siguiente_iteracion": "Alta",
        },
        {
            "tipo_de_anomalia": "Actividad inusual por usuario",
            "detectabilidad_con_dataset_actual": detectability(needs_long_history=True),
            "features_candidatas": "user_count__*, unique_admins, rare_action_count, high_risk_sequence_count",
            "dependencia_de_mayor_historico": "Alta para construir perfil de habitualidad por administrador.",
            "prioridad_siguiente_iteracion": "Alta",
        },
        {
            "tipo_de_anomalia": "Acciones desde IP o user-agent no habitual",
            "detectabilidad_con_dataset_actual": detectability(requires_ip=True, requires_user_agent=True, needs_long_history=True),
            "features_candidatas": "first_seen_ip_count, first_seen_user_agent_count, unique_ips, unique_user_agents",
            "dependencia_de_mayor_historico": "Alta; el valor crece mucho con memoria histórica por usuario.",
            "prioridad_siguiente_iteracion": "Media-Alta",
        },
        {
            "tipo_de_anomalia": "Aumento de acciones fallidas",
            "detectabilidad_con_dataset_actual": detectability(),
            "features_candidatas": "failure_count, failure_ratio, failed_admins_nunique, authentication_flag_count",
            "dependencia_de_mayor_historico": "Media; con pocas horas sigue siendo útil para bursts evidentes.",
            "prioridad_siguiente_iteracion": "Alta",
        },
        {
            "tipo_de_anomalia": "Generación anómala de tokens o API keys",
            "detectabilidad_con_dataset_actual": detectability(needs_long_history=True),
            "features_candidatas": "token_flag_count, api_key_flag_count, rare_action_count, user_count__*",
            "dependencia_de_mayor_historico": "Alta para distinguir operación rara legítima de abuso.",
            "prioridad_siguiente_iteracion": "Alta",
        },
        {
            "tipo_de_anomalia": "Desinstalación masiva de agentes o pérdida de cobertura",
            "detectabilidad_con_dataset_actual": detectability(),
            "features_candidatas": "uninstall_flag_count, unique_hostnames, assets_nunique, high_risk_action_count",
            "dependencia_de_mayor_historico": "Media; un burst fuerte puede verse incluso con ventana corta.",
            "prioridad_siguiente_iteracion": "Muy alta",
        },
        {
            "tipo_de_anomalia": "Secuencias administrativas de alto riesgo",
            "detectabilidad_con_dataset_actual": detectability(needs_long_history=True),
            "features_candidatas": "high_risk_sequence_count, sequence_signature, previous_is_failure -> success",
            "dependencia_de_mayor_historico": "Media-Alta; más sesiones observadas mejoran el baseline.",
            "prioridad_siguiente_iteracion": "Alta",
        },
        {
            "tipo_de_anomalia": "Cambios con impacto potencial de disponibilidad",
            "detectabilidad_con_dataset_actual": detectability(),
            "features_candidatas": "config_flag_count, action_center_flag_count, permissions_flag_count, high_risk_action_ratio",
            "dependencia_de_mayor_historico": "Alta para separar cambios planeados de desviaciones operativas.",
            "prioridad_siguiente_iteracion": "Muy alta",
        },
    ]

    return pd.DataFrame(rows)
