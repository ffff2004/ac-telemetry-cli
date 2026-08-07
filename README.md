# Assetto Corsa replay workspace

This repository is a uv workspace containing two related Python packages:

- [`ac-replay-parser`](packages/ac-replay-parser/README.md) parses version 16
  `.acreplay` binaries into typed objects.
- [`ac-telemetry`](packages/ac-telemetry/README.md) consumes those objects and
  produces normalized telemetry, lap, event, segment, setup, and quality
  tables.

The application flow is:

```text
.acreplay → ac-replay-parser → typed replay objects → ac-telemetry → dataset
```

Replay CSV is not an input format for `ac-telemetry`. Generated datasets may
still use CSV or Parquet storage, and individual tables can be exported as
CSV.

## Workspace setup

From the repository root:

```bash
uv sync --package ac-telemetry --extra dev
uv run --package ac-telemetry pytest packages/ac-telemetry/tests
uv run --package ac-replay-parser --extra dev pytest packages/ac-replay-parser/tests
```

See the package READMEs for their APIs, CLI usage, data model, and limitations:

- [Parser package documentation](packages/ac-replay-parser/README.md)
- [Telemetry package documentation](packages/ac-telemetry/README.md)

Example dataset and segment configurations are in [`examples/`](examples/).
