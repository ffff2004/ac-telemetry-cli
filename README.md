# ac-telemetry

A Python CLI and library that mechanically preprocesses Assetto Corsa replay-export CSV files for later GUI, statistical, or AI analysis.

It deliberately does **not** produce coaching advice. It produces normalized facts, event tables, segment passes, setup metadata, quality flags, and a compact AI context file.

## Install

```bash
python -m pip install -e .
```

For Parquet output:

```bash
python -m pip install -e '.[parquet]'
```

Without `pyarrow`, `--storage auto` falls back to CSV and records that fact in `manifest.json`.

## Inspect a replay

```bash
ac-telemetry inspect AC_300726-155603_O_ks_ferrari_f2004_spa_.csv
```

## Preprocess one or more replays with one setup

```bash
ac-telemetry preprocess \
  AC_300726-155603_O_ks_ferrari_f2004_spa_.csv \
  --setup gear+aero.ini \
  --setup-sp gear+aero.sp \
  --segments examples/spa_segments.json \
  --output build/f2004-july \
  --storage auto
```

## Preprocess sessions with different setups

Create a dataset config similar to `examples/dataset.json`, then run:

```bash
ac-telemetry preprocess \
  --config examples/dataset.json \
  --output build/f2004-history
```

Relative paths inside the config are resolved relative to the config file.

## Other commands

```bash
ac-telemetry validate build/f2004-history
ac-telemetry summarize build/f2004-history
ac-telemetry export build/f2004-history --table segments/passes --output passes.csv
ac-telemetry merge build/session-a build/session-b --output build/combined
```

## Output dataset

The exact extension is `.parquet` or `.csv`, depending on storage support.

```text
output/
├── manifest.json
├── sessions.*
├── laps.*
├── samples.*
├── events/
│   ├── braking.*
│   ├── throttle.*
│   ├── shifts.*
│   ├── lockups.*
│   └── wheelspin.*
├── segments/
│   ├── definitions.json
│   └── passes.*
├── setup/
│   ├── normalized.*
│   └── raw.json
├── summaries/
│   ├── segment_statistics.*
│   └── ai_context.json
└── quality/
    ├── flags.*
    └── validation.json
```

## Important limitations

- The supplied replay CSV does not contain AC's native normalized spline position. `progress` is therefore normalized cumulative horizontal path distance per lap and is marked `cumulative_distance_proxy`.
- Pit-lane, track-valid, tyre-temperature, direct TC torque-cut, aero-load, and differential-lock channels are not present in the replay export.
- `tc_activity_proxy` and `rear_tire_stress_proxy` are explicitly proxies, not hidden AC channels.
- The example Spa distance boundaries are calibrated to the supplied F2004 data and should not be treated as universal track coordinates.

## Library usage

```python
from pathlib import Path
from ac_telemetry import ProcessingConfig, preprocess_dataset

preprocess_dataset(
    session_specs=[
        {
            "replay": "run.csv",
            "setup": "setup.ini",
            "setup_label": "baseline",
        }
    ],
    output_dir=Path("build/run"),
    segment_path=Path("spa_segments.json"),
    config=ProcessingConfig(),
)
```
