from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .config import ProcessingConfig
from .events import detect_all_events
from .manifest import DATASET_SCHEMA_VERSION, table_manifest
from .replay import load_replay
from .segments import load_segment_definitions, segment_passes
from .setup_parser import build_setup_diffs, parse_setup_bundle
from .storage import DatasetStorage, TableRef
from .summary import build_ai_context, build_segment_statistics
from .util import json_dump, sha256_file, stable_id, utc_now_iso
from .validation import validate_dataset


def preprocess_dataset(
    session_specs: list[dict[str, Any]],
    output_dir: Path,
    segment_path: Path | None = None,
    config: ProcessingConfig | None = None,
    storage_format: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    config = config or ProcessingConfig()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = DatasetStorage(output_dir, storage_format)
    segment_definitions = load_segment_definitions(segment_path)

    all_sessions: list[dict[str, Any]] = []
    all_samples: list[pd.DataFrame] = []
    all_laps: list[pd.DataFrame] = []
    all_quality: list[pd.DataFrame] = []
    all_setups: list[pd.DataFrame] = []
    setup_metadata_by_id: dict[str, dict[str, Any]] = {}
    source_files: list[dict[str, Any]] = []

    for spec in session_specs:
        replay_path = Path(spec["replay"]).expanduser().resolve()
        setup_path = Path(spec["setup"]).expanduser().resolve() if spec.get("setup") else None
        setup_sp_path = Path(spec["setup_sp"]).expanduser().resolve() if spec.get("setup_sp") else None
        setup_meta, setup_table = parse_setup_bundle(
            setup_path, setup_sp_path, spec.get("setup_label")
        )
        setup_id = setup_meta["setup_id"] if setup_meta else None
        if setup_meta and setup_id is not None and setup_id not in setup_metadata_by_id:
            setup_metadata_by_id[setup_id] = setup_meta
            all_setups.append(setup_table)

        results = load_replay(
            replay_path,
            config,
            setup_id=setup_id,
            session_label=spec.get("session_label"),
            driver_name=spec.get("driver_name"),
        )
        for result in results:
            all_sessions.append(result.metadata)
            all_samples.append(result.samples)
            all_laps.append(result.laps)
            if not result.quality_flags.empty:
                all_quality.append(result.quality_flags)
        source_files.append(
            {
                "path": str(replay_path),
                "name": replay_path.name,
                "sha256": results[0].metadata["source_hash"],
                "type": "ac_replay",
            }
        )
        if setup_path:
            source_files.append(
                {
                    "path": str(setup_path),
                    "name": setup_path.name,
                    "sha256": sha256_file(setup_path),
                    "type": "ac_setup_ini",
                }
            )
        if setup_sp_path:
            source_files.append(
                {
                    "path": str(setup_sp_path),
                    "name": setup_sp_path.name,
                    "sha256": sha256_file(setup_sp_path),
                    "type": "ac_setup_sp",
                }
            )

    sessions = pd.DataFrame(all_sessions)
    # Store nested replay metadata as JSON-compatible string in the tabular layer.
    if "replay_metadata" in sessions:
        sessions["replay_metadata"] = sessions["replay_metadata"].map(lambda value: str(value))
    samples = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    laps = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
    quality = pd.concat(all_quality, ignore_index=True) if all_quality else pd.DataFrame(
        columns=["severity", "code", "session_id", "lap_id", "sample_start", "sample_end", "message", "affected_channels"]
    )
    setups = pd.concat(all_setups, ignore_index=True) if all_setups else pd.DataFrame()

    event_tables = detect_all_events(samples, config) if not samples.empty else {}
    # Counts become lap-level facts after event detection.
    if not laps.empty:
        count_specs = {
            "events/braking": "braking_event_count",
            "events/shifts": "shift_count",
            "events/lockups": "lockup_event_count",
            "events/wheelspin": "wheelspin_event_count",
        }
        for logical, column in count_specs.items():
            table = event_tables.get(logical, pd.DataFrame())
            counts = table.groupby("lap_id").size() if not table.empty else pd.Series(dtype=int)
            laps[column] = laps["lap_id"].map(counts).fillna(0).astype(int)

    setup_diffs = build_setup_diffs(setups)
    passes = segment_passes(samples, laps, segment_definitions)
    segment_statistics = build_segment_statistics(passes, sessions)
    ai_context = build_ai_context(sessions, laps, segment_statistics, quality)

    refs: list[TableRef] = []
    refs.append(storage.write("sessions", sessions))
    refs.append(storage.write("laps", laps))
    refs.append(storage.write("samples", samples))
    refs.append(storage.write("quality/flags", quality))
    if not setups.empty:
        refs.append(storage.write("setup/normalized", setups))
    if not setup_diffs.empty:
        refs.append(storage.write("setup/diffs", setup_diffs))
    for logical, table in event_tables.items():
        refs.append(storage.write(logical, table))
    refs.append(storage.write("segments/passes", passes))
    refs.append(storage.write("summaries/segment_statistics", segment_statistics))

    if segment_definitions is not None:
        json_dump(output_dir / "segments" / "definitions.json", segment_definitions)
    if setup_metadata_by_id:
        json_dump(output_dir / "setup" / "raw.json", setup_metadata_by_id)
    json_dump(output_dir / "summaries" / "ai_context.json", ai_context)

    dataset_id = stable_id(*(item["sha256"] for item in source_files), __version__)
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "tool_version": __version__,
        "dataset_id": dataset_id,
        "created_at": utc_now_iso(),
        "table_format": storage.format,
        "source_files": source_files,
        "processing_options": config.to_dict(),
        "progress_method": "normalized cumulative horizontal path distance per lap",
        "progress_source": "cumulative_distance_proxy",
        "segment_definition_source": str(segment_path) if segment_path else None,
        "tables": table_manifest(refs),
        "warnings": [
            "Parquet unavailable; CSV fallback used" if storage.format == "csv" else None,
            "Native AC normalized spline position is not present in parsed replay data",
        ],
    }
    manifest["warnings"] = [item for item in manifest["warnings"] if item]
    json_dump(output_dir / "manifest.json", manifest)

    validation = validate_dataset(output_dir)
    json_dump(output_dir / "quality" / "validation.json", validation)
    return manifest


def load_dataset_table(root: Path, logical_name: str) -> pd.DataFrame:
    from .util import json_load

    manifest = json_load(root / "manifest.json")
    info = manifest["tables"].get(logical_name)
    if info is None:
        raise KeyError(f"Unknown table {logical_name!r}")
    storage = DatasetStorage(root, manifest.get("table_format", "csv"))
    return storage.read(info["path"])
