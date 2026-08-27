import configparser
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import ProcessingConfig
from .sections import parse_sections_ini
from .util import sha256_file

_AI_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("stored_distance", "<f4"),
        ("stored_id", "<i4"),
    ]
)
_AI_PAYLOAD_FIELDS = (
    "ai_speed",
    "ai_throttle",
    "ai_brake",
    "ai_obsolete_lat_g",
    "ai_radius_m",
    "side_left_m",
    "side_right_m",
    "ai_camber_value",
    "ai_camber_direction",
    "ai_normal_x",
    "ai_normal_y",
    "ai_normal_z",
    "ai_segment_length_m",
    "ai_forward_x",
    "ai_forward_y",
    "ai_forward_z",
    "ai_tag",
    "ai_grade",
)
_PROJECTION_CHUNK_SIZE = 8_192
_CANDIDATE_CHUNK_SIZE = 262_144


@dataclass(frozen=True, slots=True)
class _Spline:
    source_path: Path
    source_hash: str
    closed: bool
    points: np.ndarray
    stored_distance_m: np.ndarray
    payload: np.ndarray
    distance_m: np.ndarray
    segment_start: np.ndarray
    segment_vector: np.ndarray
    segment_length_m: np.ndarray
    segment_length_squared: np.ndarray
    tangent: np.ndarray
    horizontal_tangent: np.ndarray
    left_normal: np.ndarray
    curvature_1pm: np.ndarray
    total_length_m: float
    point_tree: cKDTree = field(repr=False, compare=False)

    @property
    def segment_count(self) -> int:
        return len(self.segment_start)

    def table(self) -> pd.DataFrame:
        n = len(self.points)
        payload = self.payload
        tangent = self.tangent
        horizontal = self.horizontal_tangent
        curvature = self.curvature_1pm
        if len(tangent) < n:
            tangent = np.vstack([tangent, tangent[-1]])
            horizontal = np.vstack([horizontal, horizontal[-1]])
            curvature = np.r_[curvature, curvature[-1]]
        frame = pd.DataFrame(
            {
                "reference_index": np.arange(n, dtype=np.int64),
                "x": self.points[:, 0],
                "y": self.points[:, 1],
                "z": self.points[:, 2],
                "track_s_m": self.distance_m,
                "track_progress": self.distance_m / self.total_length_m,
                "stored_distance_m": self.stored_distance_m,
                "tangent_x": tangent[:n, 0],
                "tangent_y": tangent[:n, 1],
                "tangent_z": tangent[:n, 2],
                "heading_rad": np.arctan2(
                    horizontal[:n, 0],
                    horizontal[:n, 1],
                ),
                "curvature_1pm": curvature[:n],
            }
        )
        for index, name in enumerate(_AI_PAYLOAD_FIELDS):
            frame[name] = payload[:, index]
        return frame


def _parse_ai_spline(path: Path) -> _Spline:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"AI spline is truncated: {path}")
    version, count, _lap_time, _sample_count = struct.unpack_from("<iiii", data, 0)
    if version != 7:
        raise ValueError(f"Unsupported AI spline version {version} in {path}")
    points_end = 16 + count * _AI_POINT_DTYPE.itemsize
    if count < 2 or points_end + 4 > len(data):
        raise ValueError(f"Invalid AI spline point count {count} in {path}")
    raw = np.frombuffer(data, dtype=_AI_POINT_DTYPE, count=count, offset=16)
    extra_count = struct.unpack_from("<i", data, points_end)[0]
    if extra_count != count:
        raise ValueError(
            f"AI spline point/payload count mismatch in {path}: {count} != {extra_count}"
        )
    payload_offset = points_end + 4
    payload_bytes = count * 18 * 4
    if payload_offset + payload_bytes > len(data):
        raise ValueError(f"AI spline payload is truncated: {path}")
    payload = (
        np.frombuffer(data, dtype="<f4", count=count * 18, offset=payload_offset)
        .reshape(count, 18)
        .astype(np.float64)
    )
    points = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
    stored_distance = raw["stored_distance"].astype(np.float64)

    endpoint_gap = float(np.linalg.norm(points[0] - points[-1]))
    typical_step = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    closed = endpoint_gap <= max(50.0, 10.0 * max(typical_step, 0.1))

    starts = points if closed else points[:-1]
    ends = np.roll(points, -1, axis=0) if closed else points[1:]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    length_squared = np.einsum("ij,ij->i", vectors, vectors)
    valid = lengths > 1e-9
    if not bool(valid.all()):
        # Duplicate points are legal enough to tolerate, but they cannot define a useful
        # projection segment. Give them an epsilon length and zero tangent.
        lengths = np.where(valid, lengths, 1e-9)
    tangent = vectors / lengths[:, None]
    horizontal = tangent[:, [0, 2]]
    horizontal_norm = np.linalg.norm(horizontal, axis=1)
    horizontal = np.divide(
        horizontal,
        horizontal_norm[:, None],
        out=np.zeros_like(horizontal),
        where=horizontal_norm[:, None] > 1e-9,
    )
    left = np.stack([horizontal[:, 1], -horizontal[:, 0]], axis=1)

    # Canonical distance is rebuilt from geometry. Stored cumulative distance is kept
    # only as evidence because third-party track generators sometimes write zeros or
    # inconsistent values.
    point_distance = (
        np.concatenate([[0.0], np.cumsum(lengths[:-1])])
        if closed
        else np.concatenate([[0.0], np.cumsum(lengths)])
    )
    total = float(np.sum(lengths))

    heading = np.unwrap(np.arctan2(horizontal[:, 0], horizontal[:, 1]))
    if closed:
        next_heading = np.roll(heading, -1)
        dh = (next_heading - heading + np.pi) % (2 * np.pi) - np.pi
    else:
        dh = np.diff(heading, append=heading[-1])
    curvature = np.divide(dh, lengths, out=np.zeros_like(dh), where=lengths > 1e-6)

    return _Spline(
        source_path=path,
        source_hash=sha256_file(path),
        closed=closed,
        points=points,
        stored_distance_m=stored_distance,
        payload=payload,
        distance_m=point_distance,
        segment_start=starts,
        segment_vector=vectors,
        segment_length_m=lengths,
        segment_length_squared=length_squared,
        tangent=tangent,
        horizontal_tangent=horizontal,
        left_normal=left,
        curvature_1pm=curvature,
        total_length_m=total,
        point_tree=cKDTree(points),
    )


def _read_sections(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for section in parse_sections_ini(path.read_text(encoding="utf-8-sig")):
        if section.out_progress is None:
            raise ValueError(f"Section {section.section_id!r} has no OUT value")
        rows.append(
            {
                "section_id": section.section_id,
                "section_name": section.name,
                "start_progress": section.in_progress,
                "end_progress": section.out_progress,
            }
        )
    return rows


def _read_drs_zones(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(path, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for section in parser.sections():
        if not section.startswith("ZONE_"):
            continue
        values = parser[section]
        rows.append(
            {
                "drs_zone_id": section,
                "detection_progress": float(values["DETECTION"]),
                "start_progress": float(values["START"]),
                "end_progress": float(values["END"]),
            }
        )
    return rows


def _range_mask(values: np.ndarray, start: float, end: float) -> np.ndarray:
    if start <= end:
        return (values >= start) & (values <= end)
    return (values >= start) | (values <= end)


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def _query_global_vertices(spline: _Spline, points: np.ndarray) -> np.ndarray:
    vertex_count = len(spline.points)
    k = min(8, vertex_count)
    _distances, vertex_indices = spline.point_tree.query(
        points,
        k=k,
        workers=-1,
    )
    vertex_indices = np.asarray(vertex_indices, dtype=np.intp)
    if vertex_indices.ndim == 1:
        vertex_indices = vertex_indices.reshape(-1, k)
    return vertex_indices


def _seed_segment_candidates(
    spline: _Spline, vertex_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build sorted fixed-width incident-segment candidates for every point.

    The previous implementation built and deduplicated this topology one row at
    a time.  A sorted candidate matrix gives the same lowest-index tie-breaking
    order while allowing the geometric calculation to run in array chunks.  An
    out-of-range sentinel keeps open-spline endpoint candidates in the same
    fixed-width layout without ever projecting them.
    """
    candidates = np.concatenate((vertex_indices - 1, vertex_indices), axis=1)
    if spline.closed:
        candidates %= spline.segment_count
        valid = np.ones(candidates.shape, dtype=bool)
    else:
        valid = (candidates >= 0) & (candidates < spline.segment_count)
        candidates = np.where(valid, candidates, spline.segment_count)
    candidates.sort(axis=1)
    return candidates, candidates < spline.segment_count


def _project_seed_distances(
    spline: _Spline,
    points: np.ndarray,
    candidates: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return exact distances to the fixed-width seed candidates in chunks."""
    distances = np.empty(len(points), dtype=float)
    for start in range(0, len(points), _PROJECTION_CHUNK_SIZE):
        stop = min(start + _PROJECTION_CHUNK_SIZE, len(points))
        indices = candidates[start:stop]
        safe_indices = np.minimum(indices, spline.segment_count - 1)
        starts = spline.segment_start[safe_indices]
        vectors = spline.segment_vector[safe_indices]
        denom = spline.segment_length_squared[safe_indices]
        rel = points[start:stop, None, :] - starts
        fraction = np.divide(
            np.einsum("ijk,ijk->ij", rel, vectors),
            denom,
            out=np.zeros_like(denom),
            where=denom > 1e-12,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = starts + vectors * fraction[..., None]
        delta = points[start:stop, None, :] - projected
        distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
        distance_squared = np.where(valid[start:stop], distance_squared, np.inf)
        distances[start:stop] = np.sqrt(np.min(distance_squared, axis=1))
    return distances


def _candidate_chunk_ranges(estimated_candidates: np.ndarray) -> list[tuple[int, int]]:
    """Greedily split rows while keeping estimated candidate work in budget.

    A row larger than the budget is emitted alone; its candidates are never
    truncated.
    """
    if not len(estimated_candidates):
        return []
    chunks: list[tuple[int, int]] = []
    row_start = 0
    chunk_candidates = 0
    for row, candidate_count in enumerate(estimated_candidates):
        candidate_count = int(candidate_count)
        if candidate_count < 0:
            raise ValueError("estimated candidate counts must be non-negative")
        if (
            row > row_start
            and chunk_candidates + candidate_count > _CANDIDATE_CHUNK_SIZE
        ):
            chunks.append((row_start, row))
            row_start = row
            chunk_candidates = 0
        chunk_candidates += candidate_count
    chunks.append((row_start, len(estimated_candidates)))
    return chunks


def _project_candidate_batch(
    spline: _Spline,
    points: np.ndarray,
    rows: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project sorted, deduplicated ragged candidates and reduce by row."""
    order = np.lexsort((indices, rows))
    rows = rows[order]
    indices = indices[order]
    keep = np.r_[True, (rows[1:] != rows[:-1]) | (indices[1:] != indices[:-1])]
    rows = rows[keep]
    indices = indices[keep]

    starts = spline.segment_start[indices]
    vectors = spline.segment_vector[indices]
    denom = spline.segment_length_squared[indices]
    rel = points[rows] - starts
    fraction = np.divide(
        np.einsum("ij,ij->i", rel, vectors),
        denom,
        out=np.zeros(len(indices), dtype=float),
        where=denom > 1e-12,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    projected = starts + vectors * fraction[:, None]
    delta = points[rows] - projected
    distance_squared = np.einsum("ij,ij->i", delta, delta)

    row_starts = np.r_[0, np.flatnonzero(rows[1:] != rows[:-1]) + 1]
    row_counts = np.diff(np.r_[row_starts, len(rows)])
    minimum = np.minimum.reduceat(distance_squared, row_starts)
    best = distance_squared == np.repeat(minimum, row_counts)
    best_candidates = np.flatnonzero(best)
    first_best = np.r_[
        True,
        rows[best_candidates[1:]] != rows[best_candidates[:-1]],
    ]
    best_positions = best_candidates[first_best]
    return (
        indices[best_positions],
        fraction[best_positions],
        projected[best_positions],
        np.sqrt(minimum),
    )


def _project_track_points(
    spline: _Spline,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(points)
    segment_index = np.zeros(n, dtype=np.int64)
    fraction = np.zeros(n, dtype=float)
    projection = np.zeros((n, 3), dtype=float)
    distance = np.zeros(n, dtype=float)
    s_m = np.zeros(n, dtype=float)
    vertex_indices = _query_global_vertices(spline, points)
    seed_segments, seed_valid = _seed_segment_candidates(spline, vertex_indices)
    seed_counts = seed_valid.sum(axis=1, dtype=np.intp)
    seed_distances = _project_seed_distances(spline, points, seed_segments, seed_valid)

    # A segment whose exact distance is no greater than the seed distance has an
    # endpoint within seed_distance + segment_length. Using the longest segment
    # makes the radius query a complete candidate filter, not a distance heuristic.
    radii = seed_distances + float(np.max(spline.segment_length_m))
    radius_counts = np.asarray(
        spline.point_tree.query_ball_point(
            points,
            radii,
            workers=-1,
            return_length=True,
        ),
        dtype=np.intp,
    ).reshape(-1)
    if len(radius_counts) != n:
        raise RuntimeError(
            "cKDTree returned an unexpected radius-count shape: "
            f"expected {n}, got {len(radius_counts)}"
        )

    # A radius vertex contributes at most two incident segments. Include the
    # fixed-width seed candidates in the estimate so exact projection work stays
    # within the same budget used to bound the chunk-local ragged query.
    estimated_candidates = seed_counts + 2 * radius_counts

    # Counts are cheap to retain for the whole batch. The actual ragged query is
    # deliberately inside this loop so a dense radius cannot allocate lists for
    # every sample at once.
    for row_start, row_stop in _candidate_chunk_ranges(estimated_candidates):
        nearby_vertices = spline.point_tree.query_ball_point(
            points[row_start:row_stop],
            radii[row_start:row_stop],
            workers=-1,
        )
        row_count = row_stop - row_start
        seed_valid_chunk = seed_valid[row_start:row_stop]
        seed_counts_chunk = seed_counts[row_start:row_stop]
        seed_rows = np.repeat(np.arange(row_count, dtype=np.intp), seed_counts_chunk)
        seed_indices = seed_segments[row_start:row_stop][seed_valid_chunk]

        radius_count = int(radius_counts[row_start:row_stop].sum())
        radius_vertices: np.ndarray | None = None
        radius_rows: np.ndarray | None = None
        radius_indices: np.ndarray | None = None
        if radius_count:
            radius_vertices = np.concatenate(nearby_vertices)
            radius_rows = np.repeat(
                np.arange(row_count, dtype=np.intp),
                radius_counts[row_start:row_stop],
            )
            radius_indices = np.empty(radius_count * 2, dtype=np.intp)
            radius_indices[0::2] = radius_vertices - 1
            radius_indices[1::2] = radius_vertices
            radius_rows = np.repeat(radius_rows, 2)
            if spline.closed:
                radius_indices %= spline.segment_count
            else:
                radius_valid = (radius_indices >= 0) & (
                    radius_indices < spline.segment_count
                )
                radius_indices = radius_indices[radius_valid]
                radius_rows = radius_rows[radius_valid]
            candidate_rows = np.concatenate((seed_rows, radius_rows))
            candidate_indices = np.concatenate((seed_indices, radius_indices))
        else:
            candidate_rows = seed_rows
            candidate_indices = seed_indices

        index, fraction_chunk, projected_chunk, distance_chunk = (
            _project_candidate_batch(
                spline,
                points[row_start:row_stop],
                candidate_rows,
                candidate_indices,
            )
        )
        segment_index[row_start:row_stop] = index
        fraction[row_start:row_stop] = fraction_chunk
        projection[row_start:row_stop] = projected_chunk
        distance[row_start:row_stop] = distance_chunk
        s_m[row_start:row_stop] = (
            spline.distance_m[index] + fraction_chunk * spline.segment_length_m[index]
        )
        if spline.closed:
            s_m[row_start:row_stop] %= spline.total_length_m

        # Release the ragged query and its expanded candidate arrays before the
        # next chunk's query is materialized.
        del nearby_vertices
        del candidate_rows, candidate_indices, seed_rows, seed_indices
        if radius_count:
            del radius_vertices, radius_rows, radius_indices
    return segment_index, fraction, projection, distance, s_m


@dataclass(frozen=True, slots=True)
class TrackModel:
    """Deep module for AC-native track geometry and telemetry alignment.

    Interface invariants:
    - ``load`` accepts one concrete AC track/layout directory containing ``ai``.
    - ``align`` returns a new DataFrame; input rows/order are preserved.
    - ``track_s_m`` is canonical geometric arc length along ``fast_lane.ai`` and
      never means vehicle path length.
    - Positive ``lateral_offset_m`` is left of travel relative to the AI line.
    """

    track_dir: Path
    reference: _Spline
    pit_reference: _Spline | None
    sections: tuple[dict[str, Any], ...]
    drs_zones: tuple[dict[str, Any], ...]
    ui_metadata: dict[str, Any]

    @classmethod
    def load(cls, track_dir: Path) -> TrackModel:
        root = Path(track_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"AC track directory not found: {root}")
        fast = root / "ai" / "fast_lane.ai"
        if not fast.exists():
            fallback = root / "data" / "ideal_line.ai"
            if not fallback.exists():
                raise FileNotFoundError(
                    f"Track has neither ai/fast_lane.ai nor data/ideal_line.ai: {root}"
                )
            fast = fallback
        pit_path = root / "ai" / "pit_lane.ai"
        pit = _parse_ai_spline(pit_path) if pit_path.exists() else None
        sections = tuple(_read_sections(root / "data" / "sections.ini"))
        drs = tuple(_read_drs_zones(root / "data" / "drs_zones.ini"))
        ui_path = root / "ui" / "ui_track.json"
        ui: dict[str, Any] = {}
        if ui_path.exists():
            try:
                ui = json.loads(ui_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError, UnicodeDecodeError:
                ui = {}
        return cls(root, _parse_ai_spline(fast), pit, sections, drs, ui)

    @property
    def reference_id(self) -> str:
        return self.reference.source_hash

    def align(self, samples: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
        required = {
            "lap_id",
            "position.x",
            "position.y",
            "position.z",
            "velocity.x",
            "velocity.y",
            "velocity.z",
            "accel_world_x_ms2",
            "accel_world_y_ms2",
            "accel_world_z_ms2",
        }
        missing = sorted(required - set(samples.columns))
        if missing:
            raise ValueError(f"Samples missing track-alignment columns: {missing}")
        if samples.empty:
            return samples.copy()

        out = samples.copy()
        points = out[["position.x", "position.y", "position.z"]].to_numpy(float)
        index, fraction, projected, error, track_s = _project_track_points(
            self.reference, points
        )
        out["track_reference_index"] = index
        out["track_reference_fraction"] = fraction
        out["track_projection_x"] = projected[:, 0]
        out["track_projection_y"] = projected[:, 1]
        out["track_projection_z"] = projected[:, 2]
        out["track_projection_distance_3d_m"] = error
        out["track_s_m"] = track_s
        out["track_progress"] = track_s / self.reference.total_length_m

        horizontal_tangent = self.reference.horizontal_tangent[index]
        left = self.reference.left_normal[index]
        delta_xz = points[:, [0, 2]] - projected[:, [0, 2]]
        lateral = np.einsum("ij,ij->i", delta_xz, left)
        out["lateral_offset_m"] = lateral
        out["track_heading_rad"] = np.arctan2(
            horizontal_tangent[:, 0], horizontal_tangent[:, 1]
        )
        out["track_curvature_1pm"] = self.reference.curvature_1pm[index]

        payload = self.reference.payload
        side_left = payload[index, 5]
        side_right = payload[index, 6]
        usable_width = (
            np.isfinite(side_left)
            & np.isfinite(side_right)
            & (side_left + side_right >= config.minimum_track_width_m)
        )
        out["track_side_left_m"] = np.where(usable_width, side_left, np.nan)
        out["track_side_right_m"] = np.where(usable_width, side_right, np.nan)
        out["distance_to_left_boundary_m"] = np.where(
            usable_width, side_left - lateral, np.nan
        )
        out["distance_to_right_boundary_m"] = np.where(
            usable_width, side_right + lateral, np.nan
        )
        normalized_lateral = np.where(
            lateral >= 0,
            np.divide(
                lateral,
                side_left,
                out=np.full_like(lateral, np.nan),
                where=usable_width & (side_left > 1e-6),
            ),
            np.divide(
                lateral,
                side_right,
                out=np.full_like(lateral, np.nan),
                where=usable_width & (side_right > 1e-6),
            ),
        )
        out["lateral_position_normalized"] = normalized_lateral
        out["is_off_track_candidate"] = pd.Series(
            np.where(
                usable_width,
                (lateral > side_left + config.track_boundary_margin_m)
                | (lateral < -side_right - config.track_boundary_margin_m),
                False,
            ),
            index=out.index,
            dtype="boolean",
        )

        velocity = out[["velocity.x", "velocity.y", "velocity.z"]].to_numpy(float)
        accel = out[
            ["accel_world_x_ms2", "accel_world_y_ms2", "accel_world_z_ms2"]
        ].to_numpy(float)
        tangent = self.reference.tangent[index]
        out["velocity_along_track_ms"] = np.einsum("ij,ij->i", velocity, tangent)
        out["velocity_cross_track_ms"] = np.einsum(
            "ij,ij->i", velocity[:, [0, 2]], left
        )
        out["vertical_velocity_ms"] = velocity[:, 1]
        velocity_heading = np.arctan2(velocity[:, 0], velocity[:, 2])
        out["velocity_heading_rad"] = velocity_heading
        out["velocity_heading_error_rad"] = _wrap_angle(
            velocity_heading - out["track_heading_rad"].to_numpy(float)
        )
        along_accel = np.einsum("ij,ij->i", accel, tangent)
        cross_accel = np.einsum("ij,ij->i", accel[:, [0, 2]], left)
        out["accel_along_track_ms2"] = along_accel
        out["accel_cross_track_ms2"] = cross_accel
        out["track_long_g"] = along_accel / 9.80665
        out["track_lat_g"] = cross_accel / 9.80665

        out["track_section_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
        out["track_section_name"] = pd.Series(pd.NA, index=out.index, dtype="string")
        progress = out["track_progress"].to_numpy(float)
        for section in self.sections:
            mask = _range_mask(
                progress, section["start_progress"], section["end_progress"]
            )
            out.loc[mask, "track_section_id"] = section["section_id"]
            out.loc[mask, "track_section_name"] = section["section_name"]

        out["drs_detection_zone_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
        out["drs_activation_zone_id"] = pd.Series(
            pd.NA, index=out.index, dtype="string"
        )
        for zone in self.drs_zones:
            # Detection is a point, not an interval. Associate a narrow configurable
            # window around it so samples can be grouped without pretending it is a zone.
            detection_delta = np.abs(
                (progress - zone["detection_progress"] + 0.5) % 1.0 - 0.5
            )
            detect = (
                detection_delta * self.reference.total_length_m
                <= config.drs_detection_window_m
            )
            active = _range_mask(progress, zone["start_progress"], zone["end_progress"])
            out.loc[detect, "drs_detection_zone_id"] = zone["drs_zone_id"]
            out.loc[active, "drs_activation_zone_id"] = zone["drs_zone_id"]
        out["is_in_drs_detection_window"] = out["drs_detection_zone_id"].notna()
        out["is_in_drs_activation_zone"] = out["drs_activation_zone_id"].notna()

        if self.pit_reference is None:
            out["pit_projection_distance_3d_m"] = np.nan
            out["pit_s_m"] = np.nan
            out["pit_progress"] = np.nan
            out["is_in_pit"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        else:
            _pit_index, _pit_fraction, _pit_projection, pit_distance, pit_s = (
                _project_track_points(self.pit_reference, points)
            )
            out["pit_projection_distance_3d_m"] = np.where(
                np.isfinite(pit_distance), pit_distance, np.nan
            )
            out["pit_s_m"] = pit_s
            out["pit_progress"] = pit_s / self.pit_reference.total_length_m
            track_distance = out["track_projection_distance_3d_m"].to_numpy(float)
            in_pit = (
                np.isfinite(pit_distance)
                & (pit_distance <= config.pit_lane_max_distance_m)
                & (track_distance >= config.pit_main_reference_min_distance_m)
                & (pit_distance + config.pit_lane_preference_margin_m < track_distance)
            )
            out["is_in_pit"] = pd.Series(in_pit, index=out.index, dtype="boolean")

        # A monotonic within-lap coordinate is useful for interpolation and delta-time
        # resampling, while track_s_m remains the unmodified geometric projection.
        out["track_s_unwrapped_m"] = np.nan
        length = self.reference.total_length_m
        for _lap_id, row_index in out.groupby("lap_id", sort=False).groups.items():
            values = out.loc[row_index, "track_s_m"].to_numpy(float)
            if len(values) == 0:
                continue
            phase = np.unwrap(values / length * 2 * np.pi)
            unwrapped = phase / (2 * np.pi) * length
            out.loc[row_index, "track_s_unwrapped_m"] = unwrapped

        return out

    def tables(self) -> dict[str, pd.DataFrame]:
        tables = {"track/reference": self.reference.table()}
        if self.pit_reference is not None:
            tables["track/pit_reference"] = self.pit_reference.table()
        if self.sections:
            tables["track/sections"] = pd.DataFrame(self.sections)
        if self.drs_zones:
            tables["track/drs_zones"] = pd.DataFrame(self.drs_zones)
        return tables

    @property
    def metadata(self) -> dict[str, Any]:
        stored = self.reference.stored_distance_m
        stored_sane = bool(
            len(stored) > 1
            and np.isfinite(stored).all()
            and np.all(np.diff(stored) >= -1e-6)
            and stored[-1] > 0
            and abs(float(stored[-1]) - self.reference.total_length_m)
            <= 0.2 * self.reference.total_length_m
        )
        return {
            "track_dir": str(self.track_dir),
            "reference_source": str(self.reference.source_path),
            "track_reference_id": self.reference_id,
            "track_reference_length_m": self.reference.total_length_m,
            "reference_point_count": len(self.reference.points),
            "reference_closed": self.reference.closed,
            "stored_distance_sane": stored_sane,
            "distance_source": "geometric_reconstruction",
            "pit_reference_source": str(self.pit_reference.source_path)
            if self.pit_reference is not None
            else None,
            "pit_reference_id": self.pit_reference.source_hash
            if self.pit_reference is not None
            else None,
            "ui": self.ui_metadata,
        }
