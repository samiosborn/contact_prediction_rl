# src/contact_prediction_rl/robots/hx5_d20_contacts.py

"""Shared HX5-D20 right-hand fingertip-to-object contact supervision.

The order of output columns is always thumb, index, middle, ring, little.
Only cube-filtered forces are used; contacts with the table, palm, or other
robot links must not contribute to the privileged contact labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import ContactSensorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FINGERTIP_NAMES = ("thumb", "index", "middle", "ring", "little")
FINGERTIP_SENSOR_NAMES = tuple(f"{finger}_contact" for finger in FINGERTIP_NAMES)
CONTACT_FORCE_THRESHOLD_N = 0.1

# Preserve anatomical ordering when resolving distal links in the converted USD
_FINGERTIP_LINK_PATHS = {
    "thumb": "finger_r_link1/finger_r_link2/finger_r_link3/finger_r_link4",
    "index": "finger_r_link5/finger_r_link6/finger_r_link7/finger_r_link8",
    "middle": "finger_r_link9/finger_r_link10/finger_r_link11/finger_r_link12",
    "ring": "finger_r_link13/finger_r_link14/finger_r_link15/finger_r_link16",
    "little": "finger_r_link17/finger_r_link18/finger_r_link19/finger_r_link20",
}


def fingertip_contact_sensor_cfg(
    finger: str,
    *,
    hand_base_path: str = "{ENV_REGEX_NS}/Robot/ConvertedAsset/Geometry/hx5_d20_right_base",
    object_path: str = "{ENV_REGEX_NS}/Object",
) -> ContactSensorCfg:
    """Create one single-body sensor filtered to the task's manipulated object."""
    if finger not in _FINGERTIP_LINK_PATHS:
        raise ValueError(f"Unknown HX5-D20 fingertip {finger!r}; expected one of {FINGERTIP_NAMES}")

    return ContactSensorCfg(
        prim_path=f"{hand_base_path}/{_FINGERTIP_LINK_PATHS[finger]}",
        filter_prim_paths_expr=[object_path],
        update_period=0.0,
        force_threshold=CONTACT_FORCE_THRESHOLD_N,
    )


def fingertip_object_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return force-thresholded distal-link contacts with shape ``(num_envs, 5)``.

    Read the filtered force matrix (not net forces, which include the table).
    One sensor body and one filter body are expected for each finger. The
    0.1-N criterion is applied explicitly: the sensor's ``force_threshold``
    only governs optional air/contact-time tracking. These are link-level
    labels, with no localisation to a geometric patch.
    """
    labels = []
    for name in FINGERTIP_SENSOR_NAMES:
        force_matrix = env.scene[name].data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError(f"{name} has no filtered contact-force matrix")

        forces = force_matrix.torch if hasattr(force_matrix, "torch") else force_matrix
        expected_shape = (env.num_envs, 1, 1, 3)
        if tuple(forces.shape) != expected_shape:
            raise ValueError(f"{name}: expected filtered force shape {expected_shape}, got {tuple(forces.shape)}")

        magnitude = torch.linalg.vector_norm(forces[:, 0, 0, :], dim=-1)
        labels.append((magnitude > CONTACT_FORCE_THRESHOLD_N).to(dtype=torch.float32))

    return torch.stack(labels, dim=-1)
