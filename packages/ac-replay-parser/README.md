# ac-replay-parser

`ac-replay-parser` parses Assetto Corsa version 16 `.acreplay` binaries into
typed Python objects. It is the low-level package used by
[`ac-telemetry`](../ac-telemetry/README.md); it does not depend on pandas or
any other third-party package.

## Python API

The primary API is `parse_replay_data()`. It accepts bytes and performs no
file-system access or CSV conversion:

```python
from pathlib import Path

from ac_replay_parser import ReplayError, parse_replay_data

try:
    replay = parse_replay_data(Path("run.acreplay").read_bytes())
except ReplayError as error:
    print(f"Invalid replay: {error}")

print(replay.header.track)
for car in replay.cars:
    print(car.header.driver_name, len(car.frames))
```

The returned `ParsedReplay` contains:

- `header`: replay version, recording interval, weather, track, layout, and
  frame/car counts;
- `driver_names`: CSP driver names when available;
- `cars`: one `ParsedCar` per recorded car;
- `csp_data_offset`: the location of the optional CSP data block.

Each `ParsedCar` contains its `CarHeader`, typed `CarFrame` values, optional
CSP `ExtraCarFrame` values, and trailing per-car data. `CarFrame` exposes
position, rotation, velocity, wheel telemetry, lap timing, fuel, controls,
gear, damage, and status fields. CSP `ExtraCarFrame` provides clutch,
handbrake, wipers, turn signals, low beams, and extra options.

## Command-line interface

The CLI is a compatibility/export tool around the same parser API. It writes
one CSV per selected car:

```bash
uv run --package ac-replay-parser ac-replay-parser run.acreplay \
  --output generated/
```

For multi-car replays, output files include the driver name. A single driver
can be selected with:

```bash
uv run --package ac-replay-parser ac-replay-parser run.acreplay \
  --driver-name Alice \
  --output alice.csv
```

The `ac-replay-parser` CLI is implemented in [`cli.py`](src/ac_replay_parser/cli.py).
The binary parsing core is in [`parser.py`](src/ac_replay_parser/parser.py).

## Scope and limitations

- Only replay version 16 is currently supported.
- Truncated, malformed, or unsupported data raises `ReplayError`.
- `parse_replay_data()` is the preferred interface for applications; it does
  not write CSV files.
- The deterministic multi-car CSP fixture generator is in
  [`tests/generate_multi_car_csp_replay.py`](tests/generate_multi_car_csp_replay.py).

## Development

```bash
uv run --package ac-replay-parser --extra dev pytest packages/ac-replay-parser/tests
uv build --package ac-replay-parser
```

The translation source is
[`github.com/abchouhan/acreplay-parser`](https://github.com/abchouhan/acreplay-parser).
