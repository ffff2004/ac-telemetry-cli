"""Parse Assetto Corsa version 16 ``.acreplay`` files into typed objects.

This is a Python translation of github.com/abchouhan/acreplay-parser.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO, Iterable, Literal, overload


REPLAY_VERSION = 16
CAR_FRAME_SIZE = 256
EXTRA_FRAME_SIZE = {6: 108, 7: 108}
POSTFIX = b"__AC_SHADERS_PATCH_v1__"
DRIVER_NAME_KEY = "DRIVER_NAME="
EXT_PERCAR_PREFIX = b"EXT_PERCAR"


class ReplayError(Exception):
	"""Raised when a replay is unsupported, truncated, or malformed."""


@dataclass(frozen=True)
class Header:
	version: int
	recording_interval: float
	weather: str
	track: str
	track_config: str
	num_cars: int
	current_recording_index: int
	num_frames: int
	num_track_objects: int


@dataclass(frozen=True)
class CarHeader:
	car_id: str
	driver_name: str
	nation_code: str
	driver_team: str
	car_skin_id: str
	num_frames: int
	num_wings: int


@dataclass(frozen=True)
class Vector3:
	x: float
	y: float
	z: float


@dataclass(frozen=True)
class WheelFrame:
	static_position: Vector3
	static_rotation: Vector3
	position: Vector3
	rotation: Vector3
	angular_velocity: float
	slip_angle: float
	slip_ratio: float
	nd_slip: float
	load: float
	dirt: int


@dataclass(frozen=True)
class CarStatus:
	raw: int
	lights: bool
	horn: bool
	camera_direction: int
	gearbox_being_damaged: bool


@dataclass(frozen=True)
class CarFrame:
	position: Vector3
	rotation: Vector3
	velocity: Vector3
	wheels: tuple[WheelFrame, ...]
	rpm: float
	steer_angle: float
	bodywork_noise: float
	drivetrain_speed: float
	current_lap_time: int
	last_lap_time: int
	best_lap_time: int
	fuel: int
	fuel_per_lap: int
	gear: int
	damage_front_deformation: int
	damage_front: int
	damage_rear: int
	damage_left: int
	damage_right: int
	gas: int
	brake: int
	current_lap: int
	unknown: int
	status: CarStatus
	unknown2: int
	dirt: int
	engine_health: int
	boost: int


@dataclass(frozen=True)
class ExtraCarFrame:
	clutch: int
	handbrake: int
	wipers: int
	turn_signals: int
	low_beams: bool
	extra_options: tuple[bool, ...]
	raw_status: int


@dataclass(frozen=True)
class ParsedCar:
	header: CarHeader
	frames: tuple[CarFrame, ...]
	extra_version: int | None
	extra_frames: tuple[ExtraCarFrame, ...]
	trailing_data: bytes


@dataclass(frozen=True)
class ParsedReplay:
	header: Header
	driver_names: tuple[str, ...]
	cars: tuple[ParsedCar, ...]
	csp_data_offset: int | None


def read_exact(stream: BinaryIO, size: int) -> bytes:
	data = stream.read(size)
	if len(data) != size:
		raise ReplayError(
			f"Unexpected end of file at offset 0x{stream.tell():x} "
			f"(wanted {size} bytes, got {len(data)})"
		)
	return data


def read_u32(stream: BinaryIO) -> int:
	return struct.unpack("<I", read_exact(stream, 4))[0]


def read_string(stream: BinaryIO) -> str:
	length = read_u32(stream)
	# Assetto Corsa strings are not documented as UTF-8.  surrogateescape
	# preserves undecodable bytes so parsing can continue losslessly.
	return read_exact(stream, length).decode("utf-8", errors="surrogateescape")


def read_header(stream: BinaryIO) -> Header:
	version = read_u32(stream)
	if version != REPLAY_VERSION:
		raise ReplayError(
			f"Only version {REPLAY_VERSION} .acreplay files are supported "
			f"(found version {version})"
		)

	recording_interval = struct.unpack("<d", read_exact(stream, 8))[0]
	return Header(
		version=version,
		recording_interval=recording_interval,
		weather=read_string(stream),
		track=read_string(stream),
		track_config=read_string(stream),
		num_cars=read_u32(stream),
		current_recording_index=read_u32(stream),
		num_frames=read_u32(stream),
		num_track_objects=read_u32(stream),
	)


def read_car_header(stream: BinaryIO) -> CarHeader:
	return CarHeader(
		car_id=read_string(stream),
		driver_name=read_string(stream),
		nation_code=read_string(stream),
		driver_team=read_string(stream),
		car_skin_id=read_string(stream),
		num_frames=read_u32(stream),
		num_wings=read_u32(stream),
	)


def get_csp_data_offset(stream: BinaryIO, file_size: int) -> int | None:
	original_position = stream.tell()
	try:
		footer_size = len(POSTFIX) + 8
		if file_size < footer_size:
			return None
		stream.seek(-footer_size, 2)
		if read_exact(stream, len(POSTFIX)) != POSTFIX:
			return None
		offset, version = struct.unpack("<II", read_exact(stream, 8))
		return offset if version == 1 and offset < file_size else None
	finally:
		stream.seek(original_position)


def iter_csp_chunks(
	stream: BinaryIO, csp_offset: int, file_size: int
) -> Iterable[tuple[bytes, int]]:
	"""Yield (tag/data, payload offset) for CSP length-prefixed chunks."""
	stream.seek(csp_offset)
	footer_start = file_size - len(POSTFIX) - 8
	while stream.tell() + 4 <= footer_start:
		length = read_u32(stream)
		payload_offset = stream.tell()
		if length > footer_start - payload_offset:
			raise ReplayError(f"Malformed CSP chunk at offset 0x{payload_offset - 4:x}")
		data = read_exact(stream, length)
		yield data, payload_offset


def get_driver_names(
	stream: BinaryIO, csp_offset: int, file_size: int, num_drivers: int
) -> list[str]:
	original_position = stream.tell()
	try:
		for data, _ in iter_csp_chunks(stream, csp_offset, file_size):
			# This is the same discriminator as the C++ parser: its INI chunk is
			# the first length-prefixed CSP value larger than 255 bytes.
			if len(data) <= 255:
				continue
			ini = data.decode("utf-8", errors="surrogateescape")
			names: list[str] = []
			search_from = 0
			while len(names) < num_drivers:
				start = ini.find(DRIVER_NAME_KEY, search_from)
				if start < 0:
					break
				start += len(DRIVER_NAME_KEY)
				end = ini.find("\n", start)
				if end < 0:
					end = len(ini)
				name = ini[start:end].removesuffix("\r")
				if len(name) >= 2 and name.startswith("'") and name.endswith("'"):
					name = name[1:-1]
				names.append(name)
				search_from = start + 1
			return names + [""] * (num_drivers - len(names))
		return [""] * num_drivers
	finally:
		stream.seek(original_position)


def find_extra_frames(
	stream: BinaryIO,
	csp_offset: int,
	file_size: int,
	car_index: int,
	num_frames: int,
) -> tuple[int, bytes] | None:
	original_position = stream.tell()
	try:
		for tag, payload_offset in iter_csp_chunks(stream, csp_offset, file_size):
			if not tag.startswith(EXT_PERCAR_PREFIX):
				continue
			try:
				text = tag.decode("ascii")
				prefix, index_text = text.split(":", 1)
				version = int(prefix.rsplit("_v", 1)[1])
				extra_car_index = int(index_text)
			except (UnicodeDecodeError, ValueError, IndexError) as error:
				raise ReplayError(
					f"Malformed EXT_PERCAR tag at offset 0x{payload_offset:x}: {tag!r}"
				) from error

			if extra_car_index != car_index:
				continue
			if version not in EXTRA_FRAME_SIZE:
				return None

			compressed_size = read_u32(stream)
			compressed_data = read_exact(stream, compressed_size)
			try:
				data = zlib.decompress(compressed_data)
			except zlib.error as error:
				raise ReplayError(f"EXT_PERCAR decompression failed: {error}") from error
			expected_size = EXTRA_FRAME_SIZE[version] * num_frames
			if len(data) < expected_size:
				raise ReplayError(
					f"Truncated EXT_PERCAR data: expected at least {expected_size} bytes, "
					f"got {len(data)}"
				)
			return version, data
		return None
	finally:
		stream.seek(original_position)


def format_float(value: float, precision: int) -> str:
	return format(value, f".{precision}g")


@overload
def unpack_values(data: bytes, offset: int, fmt: Literal["e"]) -> tuple[float]: ...


@overload
def unpack_values(data: bytes, offset: int, fmt: Literal["H"]) -> tuple[int]: ...


@overload
def unpack_values(data: bytes, offset: int, fmt: Literal["eee"]) -> tuple[float, float, float]: ...


@overload
def unpack_values(data: bytes, offset: int, fmt: Literal["fff"]) -> tuple[float, float, float]: ...


@overload
def unpack_values(
	data: bytes, offset: int, fmt: Literal["eeee"]
) -> tuple[float, float, float, float]: ...


@overload
def unpack_values(data: bytes, offset: int, fmt: Literal["BBBB"]) -> tuple[int, int, int, int]: ...


@overload
def unpack_values(data: bytes, offset: int, fmt: Literal["III"]) -> tuple[int, int, int]: ...


def unpack_values(data: bytes, offset: int, fmt: str):
	return struct.unpack_from("<" + fmt, data, offset)


def yxz_as_xyz(data: bytes, offset: int) -> tuple[float, float, float]:
	y, x, z = unpack_values(data, offset, "eee")
	return x, y, z


def vector3(data: bytes, offset: int, fmt: Literal["e", "f"]) -> Vector3:
	if fmt == "e":
		x, y, z = unpack_values(data, offset, "eee")
	else:
		x, y, z = unpack_values(data, offset, "fff")
	return Vector3(x, y, z)


def yxz_vector3(data: bytes, offset: int) -> Vector3:
	x, y, z = yxz_as_xyz(data, offset)
	return Vector3(x, y, z)


def parse_car_frame(data: bytes) -> CarFrame:
	if len(data) != CAR_FRAME_SIZE:
		raise ReplayError(f"A car frame must be {CAR_FRAME_SIZE} bytes")

	static_positions = tuple(vector3(data, offset, "f") for offset in range(20, 68, 12))
	static_rotations = tuple(yxz_vector3(data, offset) for offset in range(68, 92, 6))
	positions = tuple(vector3(data, offset, "f") for offset in range(92, 140, 12))
	rotations = tuple(yxz_vector3(data, offset) for offset in range(140, 164, 6))
	angular_velocities = unpack_values(data, 172, "eeee")
	slip_angles = unpack_values(data, 180, "eeee")
	slip_ratios = unpack_values(data, 188, "eeee")
	nd_slips = unpack_values(data, 196, "eeee")
	loads = unpack_values(data, 204, "eeee")
	tire_dirt = unpack_values(data, 235, "BBBB")
	wheels = tuple(
		WheelFrame(
			static_position=static_positions[index],
			static_rotation=static_rotations[index],
			position=positions[index],
			rotation=rotations[index],
			angular_velocity=float(angular_velocities[index]),
			slip_angle=float(slip_angles[index]),
			slip_ratio=float(slip_ratios[index]),
			nd_slip=float(nd_slips[index]),
			load=float(loads[index]),
			dirt=int(tire_dirt[index]),
		)
		for index in range(4)
	)
	status_raw = int(unpack_values(data, 248, "H")[0])
	status = CarStatus(
		raw=status_raw,
		lights=bool((status_raw >> 12) & 1),
		horn=bool((status_raw >> 3) & 1),
		camera_direction=(status_raw >> 4) & 0b11,
		gearbox_being_damaged=bool((status_raw >> 9) & 1),
	)
	current_lap_time, last_lap_time, best_lap_time = unpack_values(data, 220, "III")
	return CarFrame(
		position=vector3(data, 0, "f"),
		rotation=yxz_vector3(data, 12),
		velocity=vector3(data, 164, "e"),
		wheels=wheels,
		rpm=float(unpack_values(data, 170, "e")[0]),
		steer_angle=float(unpack_values(data, 212, "e")[0]),
		bodywork_noise=float(unpack_values(data, 214, "e")[0]),
		drivetrain_speed=float(unpack_values(data, 216, "e")[0]),
		current_lap_time=int(current_lap_time),
		last_lap_time=int(last_lap_time),
		best_lap_time=int(best_lap_time),
		fuel=data[232],
		fuel_per_lap=data[233],
		gear=data[234],
		damage_front_deformation=data[239],
		damage_front=data[243],
		damage_rear=data[240],
		damage_left=data[241],
		damage_right=data[242],
		gas=data[244],
		brake=data[245],
		current_lap=data[246],
		unknown=data[247],
		status=status,
		unknown2=int(unpack_values(data, 250, "H")[0]),
		dirt=data[252],
		engine_health=data[253],
		boost=data[254],
	)


def parse_extra_frame(data: bytes, version: int) -> ExtraCarFrame:
	if len(data) != EXTRA_FRAME_SIZE.get(version):
		raise ReplayError(f"An EXT_PERCAR v{version} frame must be 108 bytes")
	if version == 6:
		status = int(unpack_values(data, 90, "H")[0])
		wipers, handbrake, clutch = data[89], data[92], data[98]
	elif version == 7:
		status = int(unpack_values(data, 88, "H")[0])
		wipers, handbrake, clutch = data[91], data[92], data[94]
	else:
		raise ReplayError(f"Unsupported EXT_PERCAR version: {version}")
	return ExtraCarFrame(
		clutch=clutch,
		handbrake=handbrake,
		wipers=wipers,
		turn_signals=status & 0b111,
		low_beams=bool((status >> 3) & 1),
		extra_options=tuple(
			bool((status >> bit) & 1) for bit in (4, 5, 6, 7, 10, 11, 12, 13, 14, 15)
		),
		raw_status=status,
	)


def read_car_frames(stream: BinaryIO, car_header: CarHeader) -> tuple[list[bytes], bytes]:
	if car_header.num_frames == 0:
		raise ReplayError("A car has zero frames")
	stream.seek(20, 1)
	frames = []
	trailing_data = b""
	for frame_index in range(car_header.num_frames):
		frames.append(read_exact(stream, CAR_FRAME_SIZE))
		if frame_index < car_header.num_frames - 1:
			stream.seek(20 + car_header.num_wings * 4, 1)
		else:
			stream.seek(car_header.num_wings * 4, 1)
			extra_count = read_u32(stream)
			if extra_count:
				trailing_data = read_exact(stream, extra_count * 8)
	return frames, trailing_data


def parse_replay_data(data: bytes | bytearray | memoryview) -> ParsedReplay:
	"""Parse complete ``.acreplay`` binary data into typed Python objects.

	The function performs no file-system access and does not write CSV output.
	``ReplayError`` is raised for unsupported, truncated, or malformed data.
	"""
	buffer = bytes(data)
	stream = BytesIO(buffer)
	file_size = len(buffer)
	header = read_header(stream)
	csp_offset = get_csp_data_offset(stream, file_size)
	csp_driver_names = (
		get_driver_names(stream, csp_offset, file_size, header.num_cars)
		if csp_offset is not None
		else None
	)

	# Two 2-byte sun angles plus 12 bytes per track object, for every frame.
	stream.seek((4 + 12 * header.num_track_objects) * header.num_frames, 1)
	parsed_cars: list[ParsedCar] = []
	for car_index in range(header.num_cars):
		if stream.tell() >= file_size:
			raise ReplayError("Attempted to read beyond file size")
		car_header = read_car_header(stream)
		raw_frames, trailing_data = read_car_frames(stream, car_header)
		extra_version: int | None = None
		extra_frames: tuple[ExtraCarFrame, ...] = ()
		if csp_offset is not None:
			raw_extra = find_extra_frames(
				stream, csp_offset, file_size, car_index, car_header.num_frames
			)
			if raw_extra is not None:
				extra_version, extra_data = raw_extra
				frame_size = EXTRA_FRAME_SIZE[extra_version]
				extra_frames = tuple(
					parse_extra_frame(extra_data[offset : offset + frame_size], extra_version)
					for offset in range(0, frame_size * car_header.num_frames, frame_size)
				)
		parsed_cars.append(
			ParsedCar(
				header=car_header,
				frames=tuple(parse_car_frame(frame) for frame in raw_frames),
				extra_version=extra_version,
				extra_frames=extra_frames,
				trailing_data=trailing_data,
			)
		)

	driver_names = (
		tuple(csp_driver_names)
		if csp_driver_names is not None
		else tuple(car.header.driver_name for car in parsed_cars)
	)
	return ParsedReplay(
		header=header,
		driver_names=driver_names,
		cars=tuple(parsed_cars),
		csp_data_offset=csp_offset,
	)
