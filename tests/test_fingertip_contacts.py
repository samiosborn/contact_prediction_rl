# tests/test_fingertip_contacts.py

"""CPU-only tests of binary, object-filtered, correctly ordered contact labels."""

from types import SimpleNamespace

import pytest
import torch

from contact_prediction_rl.robots.hx5_d20_contacts import (
    CONTACT_FORCE_THRESHOLD_N,
    FINGERTIP_NAMES,
    FINGERTIP_SENSOR_NAMES,
    fingertip_contact_sensor_cfg,
    fingertip_object_contact,
)


def _fake_env(num_envs: int = 3):
    """Simulate 5 cube-filtered contact sensors (one body and partner each)."""
    scene = {}
    for name in FINGERTIP_SENSOR_NAMES:
        forces = torch.zeros(num_envs, 1, 1, 3)
        scene[name] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=SimpleNamespace(torch=forces)))
    return SimpleNamespace(num_envs=num_envs, scene=scene)


def test_order_shape_binary_and_simultaneous_contacts():
    assert FINGERTIP_NAMES == ("thumb", "index", "middle", "ring", "little")
    env = _fake_env()
    # A scripted pair of simultaneous cube contacts in environment 0.
    env.scene["thumb_contact"].data.force_matrix_w.torch[0, 0, 0, 0] = 0.2
    env.scene["middle_contact"].data.force_matrix_w.torch[0, 0, 0, 1] = 0.3
    env.scene["little_contact"].data.force_matrix_w.torch[1, 0, 0, 2] = 0.4
    # Sub-threshold contact remains a zero bit.
    env.scene["ring_contact"].data.force_matrix_w.torch[0, 0, 0, 0] = CONTACT_FORCE_THRESHOLD_N / 2

    actual = fingertip_object_contact(env)
    expected = torch.tensor([[1, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=torch.float32)
    assert actual.shape == (3, 5)
    torch.testing.assert_close(actual, expected)
    assert torch.all((actual == 0) | (actual == 1))


def test_scripted_single_finger_contact_toggles_only_its_column():
    env = _fake_env(num_envs=2)
    for column, finger in enumerate(FINGERTIP_NAMES):
        assert torch.count_nonzero(fingertip_object_contact(env)) == 0
        force = env.scene[f"{finger}_contact"].data.force_matrix_w.torch
        force[1, 0, 0, 0] = 1.0
        labels = fingertip_object_contact(env)
        assert labels[1].tolist() == [int(i == column) for i in range(5)]
        assert labels[0].tolist() == [0, 0, 0, 0, 0]
        force.zero_()


def test_sensor_filters_cube_and_rejects_unexpected_shapes():
    env = _fake_env()
    for finger in FINGERTIP_NAMES:
        cfg = fingertip_contact_sensor_cfg(finger)
        assert cfg.filter_prim_paths_expr == ["{ENV_REGEX_NS}/Object"]
        assert cfg.update_period == 0.0

    with pytest.raises(ValueError, match="Unknown HX5-D20 fingertip"):
        fingertip_contact_sensor_cfg("palm")

    env.scene["thumb_contact"].data.force_matrix_w = None
    with pytest.raises(RuntimeError, match="thumb_contact"):
        fingertip_object_contact(env)
    env.scene["thumb_contact"].data.force_matrix_w = torch.zeros(3, 2, 1, 3)
    with pytest.raises(ValueError, match="thumb_contact"):
        fingertip_object_contact(env)
