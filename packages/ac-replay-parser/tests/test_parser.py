from __future__ import annotations

from generate_multi_car_csp_replay import make_replay

from ac_replay_parser import parse_replay_data


def test_parse_multi_car_csp_replay() -> None:
    replay = parse_replay_data(make_replay(["Alice", "Bob"], 3))

    assert replay.header.version == 16
    assert replay.header.num_cars == 2
    assert replay.driver_names == ("Alice", "Bob")
    assert [car.header.car_id for car in replay.cars] == ["fixture_car_0", "fixture_car_1"]
    assert all(len(car.frames) == 3 for car in replay.cars)
    assert [car.extra_version for car in replay.cars] == [6, 7]
    assert replay.cars[0].frames[1].position.x == 2.25
    assert replay.cars[1].extra_frames[2].clutch == 198
