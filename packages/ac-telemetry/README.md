# ac-telemetry

A Python CLI and library that mechanically preprocesses Assetto Corsa `.acreplay` files for later GUI, statistical, or AI analysis.

It deliberately does **not** produce coaching advice. It produces normalized facts, event tables, segment passes, setup metadata, quality flags, and a compact AI context file.

The commands below assume they are run from the repository root. Replay input
is `.acreplay`; generated dataset tables are stored as Parquet. The
`export-csv` command provides a CSV export for interoperability.

## Install

```bash
uv sync --package ac-telemetry --extra dev
```

## Inspect a replay

```bash
ac-telemetry inspect replay.acreplay
```

## Preprocess one or more replays with one setup

```bash
ac-telemetry preprocess \
  replay.acreplay \
  --track /path/to/assettocorsa/content/tracks/spa \
  --setup gear+aero.ini \
  --setup-sp gear+aero.sp \
  --segments examples/spa_segments.json \
  --output build/f2004-july
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
ac-telemetry export-csv build/f2004-history --table segments/passes --output passes.csv
ac-telemetry merge build/session-a build/session-b --output build/combined
```

## Output dataset

Dataset tables use the `.parquet` extension. JSON files are metadata or compact
sidecar artifacts; `export-csv` writes the selected table as CSV.

```text
output/
├── manifest.json
├── sessions.parquet
├── laps.parquet
├── samples.parquet
├── track/
│   ├── reference.parquet
│   ├── pit_reference.parquet
│   ├── sections.parquet
│   └── drs_zones.parquet
├── events/
│   ├── index.parquet
│   ├── braking.parquet
│   ├── abs_activity.parquet
│   ├── tc_activity.parquet
│   ├── throttle.parquet
│   ├── shifts.parquet
│   ├── wheel_slip.parquet
│   └── relations.parquet
├── segments/
│   ├── definitions.json
│   └── passes.parquet
├── setup/
│   ├── normalized.parquet
│   └── raw.json
├── summaries/
│   ├── segment_statistics.parquet
│   └── ai_context.json
└── quality/
    ├── flags.parquet
    └── validation.json
```

## Coordinate model and limitations

- `track_s_m` is canonical geometric arc length on AC's `ai/fast_lane.ai` (falling back to `data/ideal_line.ai`). It is independent of how far the car actually drove. `track_progress` is `track_s_m / reference_length`.
- Segment definitions are per-lap intervals and must not cross the lap boundary (`start > end`). Split such a range at `1.0`/`0.0`; `end=1.0` is the finish-line endpoint.
- `path_distance_2d_m` and `path_distance_3d_m` are the car's actual travelled path. They are deliberately separate from track position.
- Samples also contain `lateral_offset_m`, AI-line side widths/boundary distances when populated, track-relative velocity/acceleration, section annotations, DRS annotations, and pit-lane projection.
- `fast_lane.ai` is an AI racing line, not a geometric centerline. A lateral offset of zero means "on the AC AI line".
- Replay body `rotation.*` is preserved raw. The tool does not claim `rotation.y` is chassis yaw; heading and heading-rate channels derived from velocity are named accordingly.
- Pit classification uses `pit_lane.ai` only where the car is clearly separated from the main reference; the converging pit-entry/exit spline is intentionally treated as ambiguous rather than forcing a label.
- AI spline side widths are track-author data and can be imperfect on mods; `is_off_track_candidate` is therefore evidence, not an AC-native invalid-lap flag.
- `events/abs_activity` and `events/tc_activity` remain spectral intervention candidates because replay data has no native valve state or direct TC torque-cut channel.
- Event spans use half-open sample ranges and expose both `span_duration_s` and `active_duration_s`.

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
            "driven_wheels": ["rl", "rr"],
        }
    ],
    output_dir=Path("build/run"),
    track_dir=Path("/path/to/assettocorsa/content/tracks/spa"),
    segment_path=Path("examples/spa_segments.json"),
    config=ProcessingConfig(),
)
```
