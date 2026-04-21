import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "usecases" / "Cortex" / "src"))

from cortex_usecase import (  # noqa: E402
    CortexClientConfig,
    CortexManagementAuditClient,
    CortexManagementAuditParser,
    build_anomaly_taxonomy_table,
    build_field_interpretation_table,
    build_schema_profile,
    build_window_features,
    link_events_to_anomalous_windows,
    normalize_management_audit_dataframe,
    run_isolation_forest_detection,
    run_signal_review,
)


FIXTURE_PATH = REPO_ROOT / "usecases" / "Cortex" / "data" / "raw" / "sample_management_audit_logs.json"


def test_cortex_parser_reads_fixture():
    raw_df = CortexManagementAuditParser(str(FIXTURE_PATH)).parse()

    assert not raw_df.empty
    assert "AUDIT_ENTITY" in raw_df.columns
    assert "AUDIT_INSERT_TIME" in raw_df.columns


def test_cortex_feature_pipeline_runs_end_to_end():
    raw_df = CortexManagementAuditParser(str(FIXTURE_PATH)).parse()
    normalized = normalize_management_audit_dataframe(raw_df)
    features = build_window_features(normalized, time_window="15min")
    review = run_signal_review(features)
    detection = run_isolation_forest_detection(features)
    anomalous_events = link_events_to_anomalous_windows(normalized, detection["detected"], time_window="15min")

    assert not normalized.empty
    assert "event_time" in normalized.columns
    assert normalized["mentions_uninstall"].sum() >= 1
    assert normalized["is_high_risk_sequence"].sum() >= 1
    assert not features.empty
    assert "events_count" in features.columns
    assert "failure_ratio" in features.columns
    assert isinstance(review["summary"], pd.DataFrame)
    assert review["quality_report"].score > 0
    assert "iforest_label" in detection["detected"].columns
    assert (detection["detected"]["iforest_label"] == -1).sum() >= 1
    assert not anomalous_events.empty


def test_cortex_schema_and_taxonomy_tables_exist():
    raw_df = CortexManagementAuditParser(str(FIXTURE_PATH)).parse()
    normalized = normalize_management_audit_dataframe(raw_df)
    features = build_window_features(normalized, time_window="15min")
    schema_profile = build_schema_profile(raw_df)
    field_table = build_field_interpretation_table(raw_df)
    taxonomy = build_anomaly_taxonomy_table(normalized, features)

    assert not schema_profile.empty
    assert {"field", "dtype", "null_pct", "cardinality"}.issubset(schema_profile.columns)
    assert not field_table.empty
    assert {
        "campo",
        "descripcion_inferida",
        "relevancia_analitica",
        "uso_potencial",
        "fortaleza_esperada",
        "limitaciones",
    }.issubset(field_table.columns)
    assert len(taxonomy) >= 6


def test_cortex_client_config_normalizes_base_url_to_origin():
    config = CortexClientConfig(
        base_url="https://api-grupo-bancolombia.xdr.us.paloaltonetworks.com/api_keys/validate/",
        api_key_id="89",
        api_key="secret",
    )

    client = CortexManagementAuditClient(config)

    assert config.base_url == "https://api-grupo-bancolombia.xdr.us.paloaltonetworks.com"
    assert (
        client.endpoint_url
        == "https://api-grupo-bancolombia.xdr.us.paloaltonetworks.com/public_api/v1/audits/management_logs"
    )


def test_build_schema_profile_handles_unhashable_values():
    dataframe = pd.DataFrame(
        {
            "list_column": [["a", "b"], ["a", "b"], ["c"]],
            "dict_column": [{"x": 1}, {"x": 1}, {"x": 2}],
        }
    )

    profile = build_schema_profile(dataframe)

    cardinality = dict(zip(profile["field"], profile["cardinality"]))
    assert cardinality["list_column"] == 2
    assert cardinality["dict_column"] == 2
