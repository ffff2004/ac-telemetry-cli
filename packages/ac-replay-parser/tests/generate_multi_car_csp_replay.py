#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

"""Generate a deterministic multi-car CSP .acreplay fixture.

The generated replay uses the version 16 base format. Cars alternate between
CSP EXT_PERCAR versions 6 and 7, both of which are supported by this project.
"""

import argparse
import struct
import zlib
from pathlib import Path

REPLAY_VERSION = 16
CAR_FRAME_SIZE = 256
EXTRA_FRAME_SIZE = 108
POSTFIX = b"__AC_SHADERS_PATCH_v1__"


def length_prefixed(value: str | bytes) -> bytes:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<I", len(data)) + data


def make_car_frame(car_index: int, frame_index: int) -> bytes:
    """Build one 256-byte CarFrame with recognizable deterministic values."""
    data = bytearray(CAR_FRAME_SIZE)
    seed = car_index * 100 + frame_index

    # Car position and Y-X-Z half-float rotation.
    struct.pack_into("<fff", data, 0, 1.25 + seed, -2.5, 300.125)
    struct.pack_into("<eee", data, 12, 0.5, -0.25, 1.75)

    # Static wheel positions and rotations.
    for wheel_index, offset in enumerate(range(20, 68, 12)):
        struct.pack_into(
            "<fff", data, offset, seed + wheel_index + 0.1, wheel_index + 0.2, -0.3
        )
    for wheel_index, offset in enumerate(range(68, 92, 6)):
        struct.pack_into("<eee", data, offset, 0.1, 0.2 + wheel_index, 0.3)

    # Moving wheel positions and rotations.
    for wheel_index, offset in enumerate(range(92, 140, 12)):
        struct.pack_into("<fff", data, offset, wheel_index + 0.4, seed + 0.5, -0.6)
    for wheel_index, offset in enumerate(range(140, 164, 6)):
        struct.pack_into("<eee", data, offset, -0.1, -0.2 - wheel_index, -0.3)

    struct.pack_into("<eee", data, 164, 12.5 + frame_index, -0.125, 2.0)
    struct.pack_into("<e", data, 170, 7000 + seed)
    for offset in (172, 180, 188, 196, 204):
        struct.pack_into("<eeee", data, offset, 1.5, 2.5, 3.5, 4.5)
    struct.pack_into("<eee", data, 212, -15.25, 1.125, 99.5)
    struct.pack_into(
        "<III",
        data,
        220,
        1234 + frame_index,
        5678 + frame_index,
        4321,
    )

    # fuel, fuel/lap, gear, four tire-dirt values, five damage values,
    # gas, brake, lap index, unknown.
    data[232:248] = bytes(
        (200, 10, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 240, 128, frame_index, 0)
    )
    status = (1 << 12) | (1 << 9) | (1 << 3) | (2 << 4)
    struct.pack_into("<H", data, 248, status)
    struct.pack_into("<H", data, 250, 0)
    data[252:255] = bytes((11, 222, 99))  # dirt, engine health, boost
    return bytes(data)


def make_car_data(driver_name: str, car_index: int, frame_count: int) -> bytes:
    num_wings = 1
    data = bytearray()
    data += length_prefixed(f"fixture_car_{car_index}")
    data += length_prefixed(driver_name)
    data += length_prefixed("TST")
    data += length_prefixed("Fixture Team")
    data += length_prefixed(f"fixture_skin_{car_index}")
    data += struct.pack("<II", frame_count, num_wings)

    # Unknown 20-byte block before the first frame.
    data += bytes(20)
    for frame_index in range(frame_count):
        data += make_car_frame(car_index, frame_index)
        if frame_index < frame_count - 1:
            # Unknown 20-byte block and one float per wing between frames.
            data += bytes(20 + num_wings * 4)
        else:
            data += bytes(num_wings * 4)
            data += struct.pack("<I", 0)  # no trailing 8-byte values
    return bytes(data)


def make_extra_frames(version: int, car_index: int, frame_count: int) -> bytes:
    data = bytearray(EXTRA_FRAME_SIZE * frame_count)
    for frame_index in range(frame_count):
        offset = frame_index * EXTRA_FRAME_SIZE
        if version == 6:
            status_offset = offset + 90
            data[offset + 89] = 1 + frame_index % 4  # wipers
            data[offset + 92] = 70 + car_index  # handbrake
            data[offset + 98] = 200 - frame_index  # clutch
        else:
            status_offset = offset + 88
            data[offset + 91] = 1 + frame_index % 4
            data[offset + 92] = 70 + car_index
            data[offset + 94] = 200 - frame_index

        # turnSignals=car index modulo 5, low beams and options A/C/E/J on.
        status = (
            (car_index % 5) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 10) | (1 << 15)
        )
        struct.pack_into("<H", data, status_offset, status)
    return bytes(data)


def make_csp_data(driver_names: list[str], frame_count: int) -> bytes:
    ini_lines = []
    for car_index, driver_name in enumerate(driver_names):
        ini_lines.extend((f"[CAR_{car_index}]", f"DRIVER_NAME='{driver_name}'"))
    ini = ("\n".join(ini_lines) + "\n;" + " fixture-padding" * 20).encode("utf-8")
    if len(ini) <= 255:
        raise AssertionError("CSP INI chunk must be larger than 255 bytes")

    data = bytearray(length_prefixed(ini))
    for car_index in range(len(driver_names)):
        version = 6 + car_index % 2
        extra_data = make_extra_frames(version, car_index, frame_count)
        compressed = zlib.compress(extra_data)
        tag = f"EXT_PERCAR_v{version}:{car_index}"
        data += length_prefixed(tag)
        data += struct.pack("<I", len(compressed))
        data += compressed
    return bytes(data)


def make_replay(driver_names: list[str], frame_count: int) -> bytes:
    if len(driver_names) < 2:
        raise ValueError("At least two driver names are required")
    if frame_count < 1:
        raise ValueError("Frame count must be at least one")

    num_track_objects = 0
    data = bytearray(struct.pack("<Id", REPLAY_VERSION, 1000.0 / 60.0))
    data += length_prefixed("clear")
    data += length_prefixed("fixture_track")
    data += length_prefixed("fixture_layout")
    data += struct.pack(
        "<IIII",
        len(driver_names),
        frame_count,
        frame_count,
        num_track_objects,
    )
    # Two 2-byte sun angles per replay frame. There are no track objects.
    data += bytes(4 * frame_count)
    for car_index, driver_name in enumerate(driver_names):
        data += make_car_data(driver_name, car_index, frame_count)

    csp_offset = len(data)
    data += make_csp_data(driver_names, frame_count)
    data += POSTFIX
    data += struct.pack("<II", csp_offset, 1)
    return bytes(data)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic multi-car CSP .acreplay fixture"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("multi_car_csp_fixture.acreplay"),
        help="output file (default: %(default)s)",
    )
    parser.add_argument(
        "--drivers",
        nargs="+",
        default=["Alice", "Bob"],
        help="driver names; at least two are required (default: Alice Bob)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=2,
        help="frames per car (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output file if it already exists",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(
            f'Output file "{args.output}" already exists; pass --force to overwrite it'
        )

    try:
        data = make_replay(args.drivers, args.frames)
    except ValueError as error:
        raise SystemExit(f"Invalid arguments: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(
        f"Wrote {args.output} ({len(data)} bytes, {len(args.drivers)} cars, "
        f"{args.frames} frames per car)"
    )
    for car_index, driver_name in enumerate(args.drivers):
        print(f"  car {car_index}: {driver_name!r}, EXT_PERCAR v{6 + car_index % 2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
