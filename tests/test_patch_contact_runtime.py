# tests/test_patch_contact_runtime.py

"""Pure CPU tests: parallel patch labels and diagnostic-consistent decisions."""

from types import SimpleNamespace

import numpy as np
import torch

from contact_prediction_rl.robots.config import hx5_d20_patches as PATCH_CFG
from contact_prediction_rl.robots.hx5_d20_patches import (
    FINGERS,
    PatchSelection,
    _to_torch,
    active_contact_is_inside,
    classify_patch_contacts,
    patch_from_bounds,
)


def _batch():
    lo, hi = np.array([-.0095, -.0115, -.00625]), np.array([.0125, .0115, .038])
    patches = [patch_from_bounds(f, lo, hi, (i + 1,)) for i, f in enumerate(FINGERS)]
    poses = torch.zeros(2, 5, 7)
    poses[..., 6] = 1.0  # XYZW identity
    return {
        "num_envs": 2,
        "contact_geom": torch.tensor([[1, 5], [2, 5], [5, 3], [1, 6], [-1, -1]]),
        "contact_addresses": torch.tensor([[0, -1], [1, -1], [2, -1], [3, -1], [-1, -1]]),
        "contact_pos_w": torch.tensor([[.009, .002, .036], [.0076, .011, .0066], [.009, .002, .036],
                                        [.009, .002, .036], [0., 0., 0.]]),
        "contact_dist": torch.tensor([-.0002, -.0001, -.0001, -.0001, 0.]),
        "contact_world": torch.tensor([0, 0, 1, 1, -1]),
        "num_active": torch.tensor([4]),
        "geom_tags": torch.tensor([0, 1, 2, 3, 4, 5, -1]),
        "body_poses_w": poses,
        "finger_link_ids": torch.arange(5),
        "lower_local": torch.tensor(np.array([p.lower_local_m for p in patches]), dtype=torch.float32),
        "upper_local": torch.tensor(np.array([p.upper_local_m for p in patches]), dtype=torch.float32),
        "tolerance_m": torch.full((5,), PATCH_CFG.INCLUSION_TOLERANCE_M),
        "max_signed_distance_m": PATCH_CFG.MAX_SIGNED_DISTANCE_M,
    }


def test_multi_env_only_active_cube_patch_pairs_count():
    batch = _batch()
    torch.testing.assert_close(classify_patch_contacts(**batch), torch.tensor([[0., 1., 0., 0., 0.], [0., 0., 0., 1., 0.]]))


def test_contact_release_and_thresholds():
    batch = _batch()
    batch["num_active"].fill_(0)
    assert classify_patch_contacts(**batch).sum() == 0
    batch["num_active"].fill_(4)
    batch["contact_addresses"].fill_(-1)
    assert classify_patch_contacts(**batch).sum() == 0
    batch["contact_addresses"][0, 0] = 0
    batch["contact_dist"][0] = PATCH_CFG.MAX_SIGNED_DISTANCE_M + .001
    assert classify_patch_contacts(**batch).sum() == 0


def test_rotated_link_and_multiple_points_or():
    batch = _batch()
    batch["body_poses_w"][0, 1, 5:7] = torch.tensor([2**-.5, 2**-.5])
    batch["contact_pos_w"][0] = torch.tensor([-.002, .009, .036])
    batch["contact_pos_w"][1] = torch.tensor([-.002, .009, .036])
    batch["contact_geom"][1] = torch.tensor([1, 5])
    torch.testing.assert_close(classify_patch_contacts(**batch), torch.tensor([[0., 1., 0., 0., 0.], [0., 0., 0., 1., 0.]]))


def test_production_and_diagnostic_geometry_agree_on_same_world_point():
    batch = _batch()
    predictions = classify_patch_contacts(**batch)
    finger = "index"
    lo = np.array([-.0095, -.0115, -.00625])
    hi = np.array([.0125, .0115, .038])
    patch = patch_from_bounds(finger, lo, hi, (1,))
    inside, _ = active_contact_is_inside(
        patch, batch["contact_pos_w"][0].numpy(), np.zeros(3), np.array([0., 0., 0., 1.]),
        actual_pair=True, active_constraint=True,
        signed_distance_m=float(batch["contact_dist"][0]),
        max_signed_distance_m=PatchSelection().max_signed_distance_m,
    )
    assert inside == bool(predictions[0, 1])
    assert PatchSelection().inclusion_tolerance_m == PATCH_CFG.INCLUSION_TOLERANCE_M


def test_isaaclab_proxy_torch_view():
    view = torch.empty(2, 5)
    assert _to_torch(SimpleNamespace(torch=view)) is view
