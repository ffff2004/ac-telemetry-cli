from __future__ import annotations

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
    tangent: np.ndarray
    horizontal_tangent: np.ndarray
    left_normal: np.ndarray
    curvature_1pm: np.ndarray
    total_length_m: float
    point_tree: cKDTree = field(repr=False, compare=False)
    segment_midpoint_tree: cKDTree = field(repr=False, compare=False)

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
    midpoints = starts + 0.5 * vectors

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
        tangent=tangent,
        horizontal_tangent=horizontal,
        left_normal=left,
        curvature_1pm=curvature,
        total_length_m=total,
        point_tree=cKDTree(points),
        segment_midpoint_tree=cKDTree(midpoints[:, [0, 2]]),
    )


def _read_ini_ranges(path: Path, prefix: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(path, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for section in parser.sections():
        if not section.startswith(prefix):
            continue
        values = parser[section]
        if prefix == "SECTION_":
            rows.append(
                {
                    "section_id": section,
                    "section_name": values.get("TEXT", section),
                    "start_progress": float(values["IN"]),
                    "end_progress": float(values["OUT"]),
                }
            )
        elif prefix == "ZONE_":
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


def _project_candidate_segments(
    spline: _Spline, point: np.ndarray, indices: np.ndarray
) -> tuple[int, float, np.ndarray, float]:
    starts = spline.segment_start[indices]
    vectors = spline.segment_vector[indices]
    denom = np.einsum("ij,ij->i", vectors, vectors)
    rel = point - starts
    t = np.divide(
        np.einsum("ij,ij->i", rel, vectors),
        denom,
        out=np.zeros(len(indices), dtype=float),
        where=denom > 1e-12,
    )
    t = np.clip(t, 0.0, 1.0)
    projected = starts + vectors * t[:, None]
    d2 = np.einsum("ij,ij->i", point - projected, point - projected)
    best = int(np.argmin(d2))
    return int(indices[best]), float(t[best]), projected[best], float(np.sqrt(d2[best]))


def _initial_segment(spline: _Spline, point: np.ndarray) -> int:
    _distance, point_index = spline.point_tree.query(point)
    point_index = int(point_index)
    candidates = np.asarray(
        [point_index - 1, point_index],
        dtype=int,
    )
    if spline.closed:
        candidates %= spline.segment_count
    else:
        candidates = np.unique(np.clip(candidates, 0, spline.segment_count - 1))
    return _project_candidate_segments(spline, point, candidates)[0]


def _segment_window(spline: _Spline, seed: int, search_window: int) -> np.ndarray:
    indices = seed + np.arange(-2, search_window + 1, dtype=int)
    if spline.closed:
        return indices % spline.segment_count
    return indices[(indices >= 0) & (indices < spline.segment_count)]


def _project_track_points(
    spline: _Spline,
    points: np.ndarray,
    groups: np.ndarray,
    search_window: int,
    fallback_error_m: float,
    fallback_interval_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(points)
    segment_index = np.zeros(n, dtype=np.int64)
    fraction = np.zeros(n, dtype=float)
    projection = np.zeros((n, 3), dtype=float)
    distance = np.zeros(n, dtype=float)
    s_m = np.zeros(n, dtype=float)
    previous_group: object | None = None
    previous_segment = 0
    previous_error = 0.0
    samples_since_global = 0
    for row in range(n):
        point = points[row]
        group = groups[row]
        group_changed = row == 0 or group != previous_group
        if group_changed:
            previous_segment = _initial_segment(spline, point)
            samples_since_global = 0
        indices = _segment_window(spline, previous_segment, search_window)
        best, t, projected, error = _project_candidate_segments(spline, point, indices)
        should_fallback = error > fallback_error_m and (
            group_changed
            or previous_error <= fallback_error_m
            or samples_since_global >= fallback_interval_samples
        )
        if should_fallback:
            global_seed = _initial_segment(spline, point)
            global_indices = _segment_window(spline, global_seed, search_window)
            global_result = _project_candidate_segments(spline, point, global_indices)
            if global_result[3] + 1e-6 < error:
                best, t, projected, error = global_result
            samples_since_global = 0
        else:
            samples_since_global += 1
        segment_index[row] = best
        fraction[row] = t
        projection[row] = projected
        distance[row] = error
        s = float(spline.distance_m[best] + t * spline.segment_length_m[best])
        s_m[row] = s % spline.total_length_m if spline.closed else s
        previous_segment = best
        previous_group = group
        previous_error = error
    return segment_index, fraction, projection, distance, s_m


def _project_sparse_nearby(
    spline: _Spline, points: np.ndarray, cell_m: float = 25.0
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.full(len(points), np.inf, dtype=float)
    s_m = np.full(len(points), np.nan, dtype=float)
    for row, point in enumerate(points):
        # This is the radius of the square covered by the old 3x3 cell search.
        indices = spline.segment_midpoint_tree.query_ball_point(
            point[[0, 2]], cell_m * np.sqrt(8.0)
        )
        if not indices:
            continue
        best, t, _projection, error = _project_candidate_segments(
            spline, point, np.asarray(indices, dtype=int)
        )
        distance[row] = error
        value = float(spline.distance_m[best] + t * spline.segment_length_m[best])
        s_m[row] = value % spline.total_length_m if spline.closed else value
    return distance, s_m


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
        sections = tuple(_read_ini_ranges(root / "data" / "sections.ini", "SECTION_"))
        drs = tuple(_read_ini_ranges(root / "data" / "drs_zones.ini", "ZONE_"))
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
        groups = out["lap_id"].astype(str).to_numpy(object)
        index, fraction, projected, error, track_s = _project_track_points(
            self.reference,
            points,
            groups,
            config.track_projection_search_window_points,
            config.track_projection_global_fallback_error_m,
            config.track_projection_global_fallback_interval_samples,
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
            pit_distance, pit_s = _project_sparse_nearby(self.pit_reference, points)
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
