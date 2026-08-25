# ac-telemetry

A Python CLI and library that mechanically preprocesses Assetto Corsa `.acreplay` files for later GUI, statistical, or AI analysis.

It deliberately does **not** produce coaching advice. It produces normalized facts, event tables, segment passes, setup metadata, quality flags, and a compact AI context file.

The commands below assume they are run from the repository root. Replay input
is `.acreplay`; CSV is used only for optional generated table storage/export.

## Install

```bash
uv sync --package ac-telemetry --extra dev
```

For Parquet output:

```bash
uv sync --package ac-telemetry --extra parquet
```

Without `pyarrow`, `--storage auto` falls back to CSV and records that fact in `manifest.json`.

## Inspect a replay

```bash
ac-telemetry inspect replay.acreplay
```

## Preprocess one or more replays with one setup

```bash
ac-telemetry preprocess \
  replay.acreplay \
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

A multi-car replay is expanded into one session per car. Use
`--driver-name` or `driver_name` in a session config to process only one car.

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
│   ├── abs_activity.*
│   ├── tc_activity.*
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

- The parsed replay data does not contain AC's native normalized spline position. `progress` is therefore normalized cumulative horizontal path distance per lap and is marked `cumulative_distance_proxy`.
- Pit-lane, track-valid, tyre-temperature, direct TC torque-cut, aero-load, and differential-lock channels are not present in the parsed replay data.
- `events/abs_activity` contains per-wheel spectral ABS intervention candidates, not a native AC ABS-pressure or valve-state channel. Observed frequencies may be aliased by the replay sample rate.
- `events/tc_activity` contains per-rear-wheel spectral TC intervention candidates. It requires high-frequency slip-ratio activity that is absent from the throttle input and exceeds the session's non-throttle noise floor; it does not use a wheel-slip trigger threshold. AC replay data does not expose direct torque cut or the driven axle, so the detector currently assumes rear-wheel drive and observed frequencies may be aliased.
- Schema version 3 removes the inaccurate `events/throttle.tc_activity_proxy`; use `events/tc_activity` and the lap-level `tc_*` fields instead. Version 2 datasets must be regenerated before merging.
- `rear_tire_stress_proxy` is explicitly a proxy, not a hidden AC channel.
- The example Spa distance boundaries are calibrated to the supplied F2004 data and should not be treated as universal track coordinates.

## Library usage

```python
from pathlib import Path
from ac_telemetry import ProcessingConfig, preprocess_dataset

preprocess_dataset(
    session_specs=[
        {
            "replay": "run.acreplay",
            "setup": "setup.ini",
            "setup_label": "baseline",
        }
    ],
    output_dir=Path("build/run"),
    segment_path=Path("examples/spa_segments.json"),
    config=ProcessingConfig(),
)
```
