"""CPU-only XYZW orientation and snapshot-format regression tests."""

import json
import math

import pytest
import torch

from contact_prediction_rl.tasks.reorientation.grasp_snapshot import load_grasp_snapshot
from contact_prediction_rl.tasks.reorientation.orientation import axis_angle_quat, goal_orientation, orientation_error_rad, quat_rotate, signed_twist_rad


def test_known_signed_rotation_and_quaternion_sign():
    axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    initial = axis_angle_quat(torch.tensor([[1.0, 0.0, 0.0]]).expand(2, 3), torch.tensor([0.4, -0.7]))
    goal = goal_orientation(initial, axis, torch.tensor([math.pi / 12, -math.pi / 12]))
    torch.testing.assert_close(orientation_error_rad(initial, goal), torch.tensor([math.pi / 12] * 2), atol=2e-6, rtol=0)
    torch.testing.assert_close(signed_twist_rad(initial, goal, axis), torch.tensor([math.pi / 12, -math.pi / 12]), atol=1e-6, rtol=0)
    torch.testing.assert_close(orientation_error_rad(goal, -goal), torch.zeros(2), atol=1e-6, rtol=0)


def test_palm_axis_and_invalid_axis():
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    q = axis_angle_quat(axis, torch.tensor([math.pi / 2]))
    torch.testing.assert_close(quat_rotate(q, torch.tensor([[1.0, 0.0, 0.0]])), torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="axis"):
        axis_angle_quat(torch.zeros((1, 3)), torch.zeros(1))


def test_snapshot_schema(tmp_path):
    snapshot = {
        "schema_version": 1, "valid": True,
        "arm_joint_pos": {f"panda_joint{i}": 0.0 for i in range(1, 8)},
        "finger_joint_pos": {f"finger_r_joint{i}": 0.0 for i in range(1, 21)},
        "finger_target_pos": {f"finger_r_joint{i}": 0.0 for i in range(1, 21)},
        "cube_pos_w": [0.4, 0.0, 0.4], "cube_quat_w": [0.0, 0.0, 0.0, 1.0],
        "axis_world": [0.0, 0.0, 1.0],
    }
    path = tmp_path / "grasp.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    assert load_grasp_snapshot(path) == snapshot
    snapshot["finger_target_pos"].pop("finger_r_joint20")
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ValueError, match="finger_target_pos"):
        load_grasp_snapshot(path)
