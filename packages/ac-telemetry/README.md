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
  --track /path/to/assettocorsa/content/tracks/spa \
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
├── track/
│   ├── reference.*
│   ├── pit_reference.*
│   ├── sections.*
│   └── drs_zones.*
├── events/
│   ├── index.*
│   ├── braking.*
│   ├── abs_activity.*
│   ├── tc_activity.*
│   ├── throttle.*
│   ├── shifts.*
│   ├── wheel_slip.*
│   └── relations.*
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

## Coordinate model and limitations

- `track_s_m` is canonical geometric arc length on AC's `ai/fast_lane.ai` (falling back to `data/ideal_line.ai`). It is independent of how far the car actually drove. `track_progress` is `track_s_m / reference_length`.
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
