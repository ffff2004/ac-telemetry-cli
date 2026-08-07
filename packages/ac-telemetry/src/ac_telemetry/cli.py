from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProcessingConfig
from .manifest import DATASET_SCHEMA_VERSION, require_compatible_schema, table_manifest
from .pipeline import load_dataset_table, preprocess_dataset
from .replay import inspect_replay
from .storage import DatasetStorage
from .summary import build_ai_context, build_segment_statistics
from .util import json_dump, json_load
from .validation import validate_dataset


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _load_session_specs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path | None]:
    if args.config:
        config_path = Path(args.config).resolve()
        data = json_load(config_path)
        base = config_path.parent
        specs: list[dict[str, Any]] = []
        for item in data.get("sessions", []):
            spec = dict(item)
            for key in ["replay", "setup", "setup_sp"]:
                if spec.get(key):
                    path = Path(spec[key])
                    spec[key] = str(path if path.is_absolute() else base / path)
            specs.append(spec)
        segments = data.get("segments")
        segment_path = None
        if segments:
            candidate = Path(segments)
            segment_path = candidate if candidate.is_absolute() else base / candidate
        return specs, segment_path

    if not args.replays:
        raise ValueError("Provide replay files or --config")
    specs = [
        {
            "replay": replay,
            "setup": args.setup,
            "setup_sp": args.setup_sp,
            "setup_label": args.setup_label,
            "driver_name": args.driver_name,
        }
        for replay in args.replays
    ]
    return specs, Path(args.segments).resolve() if args.segments else None


def command_inspect(args: argparse.Namespace) -> int:
    results = [inspect_replay(Path(path).resolve()) for path in args.replays]
    _print_json(results[0] if len(results) == 1 else results)
    return 0


def command_preprocess(args: argparse.Namespace) -> int:
    specs, config_segments = _load_session_specs(args)
    segment_path = Path(args.segments).resolve() if args.segments else config_segments
    manifest = preprocess_dataset(
        specs,
        Path(args.output).resolve(),
        segment_path=segment_path,
        config=ProcessingConfig(),
        storage_format=args.storage,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "status": "ok",
            "dataset_id": manifest["dataset_id"],
            "output": str(Path(args.output).resolve()),
            "table_format": manifest["table_format"],
            "tables": manifest["tables"],
            "warnings": manifest["warnings"],
        }
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    result = validate_dataset(Path(args.dataset).resolve())
    _print_json(result)
    return 0 if result["status"] in {"ok", "warning"} else 2


def command_summarize(args: argparse.Namespace) -> int:
    root = Path(args.dataset).resolve()
    sessions = load_dataset_table(root, "sessions")
    laps = load_dataset_table(root, "laps")
    passes = load_dataset_table(root, "segments/passes")
    try:
        quality = load_dataset_table(root, "quality/flags")
    except KeyError:
        quality = pd.DataFrame()
    stats = build_segment_statistics(passes, sessions)
    context = build_ai_context(sessions, laps, stats, quality)
    json_dump(root / "summaries" / "ai_context.json", context)
    _print_json(context)
    return 0


def command_export(args: argparse.Namespace) -> int:
    frame = load_dataset_table(Path(args.dataset).resolve(), args.table)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    _print_json({"status": "ok", "rows": len(frame), "output": str(output)})
    return 0


def command_merge(args: argparse.Namespace) -> int:
    roots = [Path(item).resolve() for item in args.datasets]
    output = Path(args.output).resolve()
    manifests = [json_load(root / "manifest.json") for root in roots]
    require_compatible_schema(manifests)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    logical_names = sorted(set.intersection(*(set(m["tables"]) for m in manifests)))
    storage = DatasetStorage(output, args.storage)
    refs = []
    for logical in logical_names:
        frames = [load_dataset_table(root, logical) for root in roots]
        merged = pd.concat(frames, ignore_index=True).drop_duplicates()
        refs.append(storage.write(logical, merged))
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "tool_version": manifests[0].get("tool_version"),
        "dataset_id": "merged-" + "-".join(m["dataset_id"][:8] for m in manifests),
        "table_format": storage.format,
        "source_datasets": [str(root) for root in roots],
        "tables": table_manifest(refs),
        "warnings": ["Only tables common to every input dataset were merged"],
    }
    json_dump(output / "manifest.json", manifest)
    json_dump(output / "quality" / "validation.json", validate_dataset(output))
    _print_json({"status": "ok", "output": str(output), "tables": manifest["tables"]})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ac-telemetry",
        description="Mechanical preprocessing for Assetto Corsa .acreplay telemetry files.",
    )
    parser.add_argument("--version", action="version", version="ac-telemetry 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect", help="Inspect .acreplay metadata and cars")
    inspect_cmd.add_argument("replays", nargs="+")
    inspect_cmd.set_defaults(func=command_inspect)

    preprocess = sub.add_parser("preprocess", help="Build a normalized analysis dataset")
    preprocess.add_argument("replays", nargs="*")
    preprocess.add_argument("--config", help="Dataset JSON containing per-session replay/setup mappings")
    preprocess.add_argument("--setup")
    preprocess.add_argument("--setup-sp")
    preprocess.add_argument("--setup-label")
    preprocess.add_argument("--driver-name", help="Process only the named driver's car")
    preprocess.add_argument("--segments")
    preprocess.add_argument("--output", required=True)
    preprocess.add_argument("--storage", choices=["auto", "parquet", "csv"], default="auto")
    preprocess.add_argument("--overwrite", action="store_true")
    preprocess.set_defaults(func=command_preprocess)

    validate = sub.add_parser("validate", help="Validate a generated dataset")
    validate.add_argument("dataset")
    validate.set_defaults(func=command_validate)

    summarize = sub.add_parser("summarize", help="Regenerate compact AI/statistical summaries")
    summarize.add_argument("dataset")
    summarize.set_defaults(func=command_summarize)

    export = sub.add_parser("export", help="Export one logical table to CSV")
    export.add_argument("dataset")
    export.add_argument("--table", required=True, help="Logical table name, e.g. laps or segments/passes")
    export.add_argument("--output", required=True)
    export.set_defaults(func=command_export)

    merge = sub.add_parser("merge", help="Merge compatible generated datasets")
    merge.add_argument("datasets", nargs="+")
    merge.add_argument("--output", required=True)
    merge.add_argument("--storage", choices=["auto", "parquet", "csv"], default="auto")
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(func=command_merge)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
