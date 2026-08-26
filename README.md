# Assetto Corsa replay workspace

This repository is a uv workspace containing two related Python packages:

- [`ac-replay-parser`](packages/ac-replay-parser/README.md) parses version 16
  `.acreplay` binaries into typed objects.
- [`ac-telemetry`](packages/ac-telemetry/README.md) consumes those objects and
  produces normalized telemetry, lap, event, segment, setup, and quality
  tables.

The application flow is:

```text
.acreplay ─→ ac-replay-parser ─→ typed replay objects ─┐
                                                       ├→ ac-telemetry → dataset
AC track ─→ fast_lane.ai / pit_lane.ai ─→ TrackModel ─┘
```

`TrackModel` supplies the canonical circuit coordinate. Vehicle path distance is
kept separately and is never used as a proxy for track position.

Replay CSV is not an input format for `ac-telemetry`. Generated datasets may
still use CSV or Parquet storage, and individual tables can be exported as
CSV.

## Workspace setup

From the repository root:

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type pre-push --install-hooks
uv run pre-commit run --all-files
uv run pytest packages/ac-replay-parser/tests packages/ac-telemetry/tests
uv build --all-packages
```

The workspace requires Python 3.14. Commit hooks format and lint Python files
and run pyright; push hooks run the full test suite and build both packages.

See the package READMEs for their APIs, CLI usage, data model, and limitations:

- [Parser package documentation](packages/ac-replay-parser/README.md)
- [Telemetry package documentation](packages/ac-telemetry/README.md)

Example dataset and segment configurations are in [`examples/`](examples/).
