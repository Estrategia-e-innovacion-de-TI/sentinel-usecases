# Cortex Management Plane Audit Logs

Caso de uso exploratorio para analizar los management audit logs de Cortex XDR usando Sentinel como base de validación de señal, agregación y detección de anomalías administrativas con posible impacto en disponibilidad, cobertura y estabilidad operativa.

## Objetivo

Este caso de uso busca responder, con una primera iteración incremental:

- qué campos expone realmente el endpoint oficial de management audit logs
- cuáles son los campos más útiles para analítica de seguridad operacional
- qué anomalías administrativas parecen detectables con Sentinel
- qué señales pueden anticipar degradación de cobertura, disponibilidad o estabilidad

El endpoint objetivo es el oficial de Cortex XDR para management audit logs:

- `POST /public_api/v1/audits/management_logs`

## Estructura

```text
usecases/Cortex/
├── README.md
├── data/
│   ├── raw/
│   │   └── sample_management_audit_logs.json
│   └── processed/
├── notebooks/
│   └── cortex_management_plane_exploration.ipynb
├── outputs/
│   └── figures/
└── src/
    └── cortex_usecase/
        ├── __init__.py
        ├── cortex_client.py
        ├── cortex_parser.py
        ├── feature_engineering.py
        └── utils.py
```

## Prerrequisitos

Desde la raíz del repo:

```bash
pip install -e ".[dev]"
```

Opcional, si quieres visualización interactiva o RRCF:

```bash
pip install -e ".[viz,rrcf]"
```

## Variables de entorno

Requeridas para extracción real:

```bash
export XDR_BASE_URL="https://api-{tu-fqdn}"
export XDR_API_KEY_ID="1234"
export XDR_API_KEY="..."
```

Opcionales:

```bash
export XDR_AUTH_MODE="advanced"   # advanced | standard
export XDR_VERIFY_SSL="true"
export XDR_TIMEOUT_SECONDS="30"
export XDR_PAGE_SIZE="100"
export XDR_REQUEST_INTERVAL_SECONDS="0.12"
```

## Exploración rápida

Abre el notebook principal:

```bash
jupyter lab usecases/Cortex/notebooks/cortex_management_plane_exploration.ipynb
```

El notebook está preparado para empezar en modo incremental:

- `FAST_TEST_MODE = True`
- `LOOKBACK_HOURS = 6`
- `START_TIME = None`
- `END_TIME = None`
- `INITIAL_OFFSET = 0`
- `MAX_RECORDS = 100`
- `TIME_WINDOW = "15min"`

Si no encuentra credenciales en el entorno, usa el fixture `sample_management_audit_logs.json` solo para dry run técnico del parser, feature engineering y Sentinel. Ese fallback no reemplaza la validación real del esquema ni de autenticación.

## Cómo ampliar la ventana temporal

Para pasar de validación rápida a exploración más fuerte:

- aumenta `LOOKBACK_HOURS`
- o define `START_TIME` y `END_TIME` en formato ISO 8601
- incrementa `MAX_RECORDS` de forma gradual
- conserva `page_size <= 100`, porque el endpoint documenta ese máximo

Recomendación práctica:

1. validar primero con 2 a 6 horas
2. subir a 24 horas
3. luego evaluar 7 a 14 días para baselines por usuario, IP y secuencia

## Outputs esperados

Cuando se ejecuta el notebook, persiste:

- raw JSON combinado de la extracción
- metadatos de extracción
- CSV procesado a nivel evento
- CSV de features por ventana
- figuras en `usecases/Cortex/outputs/figures/`

## Notas de implementación

- `cortex_client.py` soporta autenticación estándar y avanzada, paginación, errores HTTP claros y conversión a DataFrame.
- `cortex_parser.py` reutiliza el estilo de ingestión de Sentinel mediante `BaseLogParser`.
- `feature_engineering.py` crea features por ventana, taxonomía de anomalías, tablas finales y primera integración con `SignalDiagnostics`, `RollingAggregator`, `StringAggregator`, `IsolationForestDetector` y `RRCFDetector` opcional.
- `sentinel.visualization.AnomalyVisualizer` quedó utilizable sin `plotly` ni `shap`, de modo que los gráficos estáticos funcionen aun sin extras.
