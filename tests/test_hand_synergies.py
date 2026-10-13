# tests/test_hand_synergies.py

"""Tests for the hand synergy postures."""

import re

import pytest

from contact_prediction_rl.robots.franka_hx5d20 import (
    FRANKA_HX5D20_HAND_CLOSED,
    FRANKA_HX5D20_HAND_CLOSED_THUMB_OPEN,
    FRANKA_HX5D20_HAND_SYNERGIES,
    resolve_joint_posture,
)

# lower and upper limits [rad] of finger_r_joint1 ... finger_r_joint20 in the HX5-D20 URDF
THUMB_LIMITS = [(-1.57, 1.57), (-1.57, 0.0), (0.0, 1.57), (0.0, 1.57)]
FINGER_LIMITS = [(-0.6, 0.6), (0.0, 2.0), (0.0, 1.57), (0.0, 1.57)]
JOINT_LIMITS = THUMB_LIMITS + 4 * FINGER_LIMITS
JOINT_NAMES = [f"finger_r_joint{i}" for i in range(1, 21)]


def posture(name: str) -> list[float]:
    return resolve_joint_posture(FRANKA_HX5D20_HAND_SYNERGIES[name], JOINT_NAMES)


def test_curl_and_thumb_make_the_closed_posture():
    closed = resolve_joint_posture(FRANKA_HX5D20_HAND_CLOSED, JOINT_NAMES)
    combined = [c + t for c, t in zip(posture("curl"), posture("thumb"))]

    assert combined == pytest.approx(closed)


def test_synergies_act_on_separate_joints():
    for first, second in [("curl", "thumb"), ("curl", "spread"), ("thumb", "spread")]:
        moved = [a != 0.0 and b != 0.0 for a, b in zip(posture(first), posture(second))]

        assert not any(moved), f"{first} and {second} move the same joint"


def test_synergies_stay_inside_the_joint_limits_at_full_closure():
    for name in FRANKA_HX5D20_HAND_SYNERGIES:
        for value, (lower, upper) in zip(posture(name), JOINT_LIMITS):
            # the thumb opposition of the closed posture (1.571) is 1 mrad past the rounded URDF limit
            assert lower - 2e-3 <= value <= upper + 2e-3


def test_spread_moves_the_outer_fingers_apart():
    spread = posture("spread")
    index, middle, ring, little = (spread[i] for i in (4, 8, 12, 16))

    assert index < middle < 0.0 < ring < little
    assert index == pytest.approx(-little)


def test_thumb_open_posture_is_the_closed_posture_without_the_thumb():
    closed = resolve_joint_posture(FRANKA_HX5D20_HAND_CLOSED, JOINT_NAMES)
    thumb_open = resolve_joint_posture(FRANKA_HX5D20_HAND_CLOSED_THUMB_OPEN, JOINT_NAMES)

    assert thumb_open[:4] == [0.0] * 4
    assert thumb_open[4:] == pytest.approx(closed[4:])
    assert thumb_open == pytest.approx(posture("curl"))


def test_thumb_open_posture_names_every_finger_joint():
    # the binary hand action raises an error unless every joint it controls has a value
    for name in JOINT_NAMES:
        matches = [p for p in FRANKA_HX5D20_HAND_CLOSED_THUMB_OPEN if re.fullmatch(p, name)]

        assert len(matches) == 1, name
