"""Command-line interface for converting Assetto Corsa replays to CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from .parser import CarFrame, ExtraCarFrame, Header, ParsedCar, ReplayError, parse_replay_data

CSV_FORMAT_VERSION = 2

CAR_FRAME_LABELS = (
	"frame,"
	"position.x,position.y,position.z,"
	"rotation.x,rotation.y,rotation.z,"
	"velocity.x,velocity.y,velocity.z,"
	"wheelFL.staticPosition.x,wheelFL.staticPosition.y,wheelFL.staticPosition.z,"
	"wheelFR.staticPosition.x,wheelFR.staticPosition.y,wheelFR.staticPosition.z,"
	"wheelRL.staticPosition.x,wheelRL.staticPosition.y,wheelRL.staticPosition.z,"
	"wheelRR.staticPosition.x,wheelRR.staticPosition.y,wheelRR.staticPosition.z,"
	"wheelFL.staticRotation.x,wheelFL.staticRotation.y,wheelFL.staticRotation.z,"
	"wheelFR.staticRotation.x,wheelFR.staticRotation.y,wheelFR.staticRotation.z,"
	"wheelRL.staticRotation.x,wheelRL.staticRotation.y,wheelRL.staticRotation.z,"
	"wheelRR.staticRotation.x,wheelRR.staticRotation.y,wheelRR.staticRotation.z,"
	"wheelFL.position.x,wheelFL.position.y,wheelFL.position.z,"
	"wheelFR.position.x,wheelFR.position.y,wheelFR.position.z,"
	"wheelRL.position.x,wheelRL.position.y,wheelRL.position.z,"
	"wheelRR.position.x,wheelRR.position.y,wheelRR.position.z,"
	"wheelFL.rotation.x,wheelFL.rotation.y,wheelFL.rotation.z,"
	"wheelFR.rotation.x,wheelFR.rotation.y,wheelFR.rotation.z,"
	"wheelRL.rotation.x,wheelRL.rotation.y,wheelRL.rotation.z,"
	"wheelRR.rotation.x,wheelRR.rotation.y,wheelRR.rotation.z,"
	"wheelFL.angularVelocity,wheelFR.angularVelocity,wheelRL.angularVelocity,wheelRR.angularVelocity,"
	"wheelFL.slipAngle,wheelFR.slipAngle,wheelRL.slipAngle,wheelRR.slipAngle,"
	"wheelFL.slipRatio,wheelFR.slipRatio,wheelRL.slipRatio,wheelRR.slipRatio,"
	"wheelFL.ndSlip,wheelFR.ndSlip,wheelRL.ndSlip,wheelRR.ndSlip,"
	"wheelFL.load,wheelFR.load,wheelRL.load,wheelRR.load,"
	"wheelFL.dirt,wheelFR.dirt,wheelRL.dirt,wheelRR.dirt,"
	"steerAngle,bodyworkNoise,drivetrainSpeed,"
	"currentLap,currentLapTime,lastLapTime,bestLapTime,"
	"fuel,fuelPerLap,rpm,gear,gas,brake,boost,"
	"damageFrontDeformation,damageFront,damageRear,damageLeft,damageRight,"
	"lights,horn,cameraDir,engineHealth,gearboxBeingDamaged,dirt"
)

EXTRA_FRAME_LABELS = (
	"clutch,handbrake,wipers,turnSignals,lowBeams,"
	"extraOptionA,extraOptionB,extraOptionC,extraOptionD,extraOptionE,"
	"extraOptionF,extraOptionG,extraOptionH,extraOptionI,extraOptionJ"
)


def format_float(value: float, precision: int) -> str:
	return format(value, f".{precision}g")


def output_car_frame(frame: CarFrame) -> str:
	values: list[str] = []

	def add_f32(items: Iterable[object]) -> None:
		values.extend(format_float(float(item), 9) for item in items)

	def add_f16(items: Iterable[object]) -> None:
		values.extend(format_float(float(item), 5) for item in items)

	def add_int(items: Iterable[object]) -> None:
		values.extend(str(int(item)) for item in items)

	add_f32((frame.position.x, frame.position.y, frame.position.z))
	add_f16((frame.rotation.x, frame.rotation.y, frame.rotation.z))
	add_f16((frame.velocity.x, frame.velocity.y, frame.velocity.z))
	for wheel in frame.wheels:
		add_f32((wheel.static_position.x, wheel.static_position.y, wheel.static_position.z))
	for wheel in frame.wheels:
		add_f16((wheel.static_rotation.x, wheel.static_rotation.y, wheel.static_rotation.z))
	for wheel in frame.wheels:
		add_f32((wheel.position.x, wheel.position.y, wheel.position.z))
	for wheel in frame.wheels:
		add_f16((wheel.rotation.x, wheel.rotation.y, wheel.rotation.z))
	add_f16(wheel.angular_velocity for wheel in frame.wheels)
	add_f16(wheel.slip_angle for wheel in frame.wheels)
	add_f16(wheel.slip_ratio for wheel in frame.wheels)
	add_f16(wheel.nd_slip for wheel in frame.wheels)
	add_f16(wheel.load for wheel in frame.wheels)
	add_int(wheel.dirt for wheel in frame.wheels)
	add_f16((frame.steer_angle, frame.bodywork_noise, frame.drivetrain_speed))
	add_int((frame.current_lap, frame.current_lap_time, frame.last_lap_time, frame.best_lap_time))
	add_int((frame.fuel, frame.fuel_per_lap))
	add_f16((frame.rpm,))
	add_int((frame.gear, frame.gas, frame.brake, frame.boost))
	add_int(
		(
			frame.damage_front_deformation,
			frame.damage_front,
			frame.damage_rear,
			frame.damage_left,
			frame.damage_right,
		)
	)
	values.extend(
		(
			str(frame.status.lights).lower(),
			str(frame.status.horn).lower(),
			str(frame.status.camera_direction),
			str(frame.engine_health),
			str(frame.status.gearbox_being_damaged).lower(),
			str(frame.dirt),
		)
	)
	return ",".join(values)


def output_extra_frame(frame: ExtraCarFrame) -> str:
	values = [
		str(frame.clutch),
		str(frame.handbrake),
		str(frame.wipers),
		str(frame.turn_signals),
		str(frame.low_beams).lower(),
	]
	values.extend(str(value).lower() for value in frame.extra_options)
	return ",".join(values)


def unique_path(path: Path) -> Path:
	if not path.exists():
		return path
	index = 2
	while True:
		candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
		if not candidate.exists():
			return candidate
		index += 1


def output_path_for_car(
	input_path: Path,
	preferred_output: str | None,
	driver_name: str,
	num_cars: int,
	target_driver_name: str | None,
) -> tuple[Path, bool]:
	if preferred_output is None or preferred_output.endswith(("/", "\\")):
		directory = Path(preferred_output) if preferred_output else Path.cwd()
		suffix = f"_{driver_name}" if target_driver_name is None and num_cars > 1 else ""
		return unique_path(directory / f"{input_path.stem}{suffix}.csv"), False

	path = Path(preferred_output)
	if target_driver_name is None and num_cars > 1:
		path = path.with_name(f"{path.stem}_{driver_name}{path.suffix}")
	if not path.suffix:
		path = path.with_suffix(".csv")
	return path, path.exists()


def write_csv(
	output_path: Path,
	header: Header,
	car: ParsedCar,
) -> None:
	car_header = car.header
	metadata = (
		("formatVersion", CSV_FORMAT_VERSION),
		("numFrames", car_header.num_frames),
		("carNumFrames", car_header.num_frames),
		("recordingInterval", format(header.recording_interval, ".6g")),
		("replayVersion", header.version),
		("weather", header.weather),
		("track", header.track),
		("trackConfig", header.track_config),
		("numCars", header.num_cars),
		("currentRecordingIndex", header.current_recording_index),
		("replayNumFrames", header.num_frames),
		("numTrackObjects", header.num_track_objects),
		("carID", car_header.car_id),
		("driverName", car_header.driver_name),
		("nationCode", car_header.nation_code),
		("driverTeam", car_header.driver_team),
		("carSkinID", car_header.car_skin_id),
		("numWings", car_header.num_wings),
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8", errors="surrogateescape", newline="\n") as output:
		for key, value in metadata:
			output.write(f"# {key} {value}\n")
		output.write(CAR_FRAME_LABELS)
		if car.extra_frames:
			output.write("," + EXTRA_FRAME_LABELS)
		output.write("\n")

		for frame_index, frame in enumerate(car.frames):
			row = f"{frame_index},{output_car_frame(frame)}"
			if car.extra_frames:
				row += "," + output_extra_frame(car.extra_frames[frame_index])
			output.write(row + "\n")


def parse_replay(
	input_path: Path,
	preferred_output: str | None = None,
	target_driver_name: str | None = None,
) -> list[Path]:
	if not input_path.is_file():
		raise ReplayError(f'File "{input_path}" not found')

	binary_data = input_path.read_bytes()
	print(f"{input_path}\n{len(binary_data)} bytes")
	replay = parse_replay_data(binary_data)
	header = replay.header
	print(
		f"Version: {header.version}\n"
		f"Recording Interval: {header.recording_interval} ms\n"
		f"Weather: {header.weather}\n"
		f"Track: {header.track}\n"
		f"Track Config: {header.track_config}\n"
		f"Number of Cars: {header.num_cars}\n"
		f"Number of Frames: {header.num_frames}"
	)
	if replay.csp_data_offset is not None:
		print("Driver Names:")
		for name in replay.driver_names:
			selected = "\t<< SELECTED" if target_driver_name == name else ""
			print(f"\t{name}{selected}")
		if target_driver_name is not None and target_driver_name not in replay.driver_names:
			raise ReplayError(f'Driver "{target_driver_name}" was not found')

	outputs: list[Path] = []
	for car in replay.cars:
		car_header = car.header
		if target_driver_name is not None and target_driver_name != car_header.driver_name:
			continue

		print(
			f"\nCar ID: {car_header.car_id}\n"
			f"Driver Name: {car_header.driver_name}\n"
			f"Nation Code: {car_header.nation_code}\n"
			f"Driver Team: {car_header.driver_team}\n"
			f"Car Skin ID: {car_header.car_skin_id}\n"
			f"Number of Frames: {car_header.num_frames}\n"
			f"Number of Wings: {car_header.num_wings}"
		)
		if car.trailing_data:
			print(f"Extra trailing bytes: {len(car.trailing_data) // 8}")
		if car.extra_version is not None:
			print(f"EXT_PERCAR version: {car.extra_version}")

		output_path, overwrites = output_path_for_car(
			input_path,
			preferred_output,
			car_header.driver_name,
			header.num_cars,
			target_driver_name,
		)
		if overwrites:
			print(f'\nFile "{output_path}" will be overwritten\n')
		write_csv(output_path, header, car)
		outputs.append(output_path)
		print(f"{output_path}\nDone!")

	return outputs


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="ac_replay_parser.py",
		description="Assetto Corsa Replay Parser (Python translation)",
	)
	parser.add_argument("inputs", metavar="INPUT", nargs="+", type=Path)
	parser.add_argument(
		"-o",
		"--output",
		help="output path with optional filename; a trailing slash denotes a directory",
	)
	parser.add_argument("--driver-name", help="only parse the named driver's car")
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_argument_parser().parse_args(argv)
	exit_code = 0
	for index, input_path in enumerate(args.inputs):
		if index:
			print("-----------------------------")
		try:
			parse_replay(input_path, args.output, args.driver_name)
		except (OSError, ReplayError) as error:
			print(f"Error: {error}", file=sys.stderr)
			exit_code = 1
	return exit_code


if __name__ == "__main__":
	raise SystemExit(main())
