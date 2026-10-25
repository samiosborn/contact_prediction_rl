"""Canonical reference path and lossless snapshot promotion are CPU-testable."""

import json
from pathlib import Path

import pytest

from contact_prediction_rl.tasks.reorientation.config.paths import GENERATED_GRASP_PATH, REFERENCE_GRASP_PATH
from contact_prediction_rl.tasks.reorientation.grasp_snapshot import grasp_snapshot_sha256, promote_grasp_snapshot


def _snapshot() -> dict:
    return {
        "schema_version": 1,
        "valid": True,
        "arm_joint_pos": {f"panda_joint{i}": 0.1 for i in range(1, 8)},
        "finger_joint_pos": {f"finger_r_joint{i}": 0.1 for i in range(1, 21)},
        "finger_target_pos": {f"finger_r_joint{i}": 0.2 for i in range(1, 21)},
        "cube_pos_w": [0.0, 0.0, 0.1],
        "cube_quat_w": [0.0, 0.0, 0.0, 1.0],
        "axis_world": [0.0, 0.0, 1.0],
    }


def test_reference_and_generated_paths_are_separate():
    assert REFERENCE_GRASP_PATH.parts[-4:] == ("data", "reorientation", "grasps", "hx5_d20_palm_up_v1.json")
    assert GENERATED_GRASP_PATH.parts[-3:] == ("runs", "reorientation", "grasp_snapshot.json")
    assert REFERENCE_GRASP_PATH.is_absolute()
    assert REFERENCE_GRASP_PATH != GENERATED_GRASP_PATH


def test_promote_preserves_exact_bytes_and_never_overwrites(tmp_path: Path):
    source = tmp_path / "generated.json"
    target = tmp_path / "reference" / "hx5_d20_palm_up_v1.json"
    source.write_text(json.dumps(_snapshot(), indent=3) + "\n")
    digest = promote_grasp_snapshot(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert digest == grasp_snapshot_sha256(source) == grasp_snapshot_sha256(target)
    assert promote_grasp_snapshot(source, target) == digest
    source.write_text(json.dumps({**_snapshot(), "candidate": "different"}) + "\n")
    with pytest.raises(FileExistsError):
        promote_grasp_snapshot(source, target)


def test_bad_grasp_is_not_promoted(tmp_path: Path):
    source = tmp_path / "invalid.json"
    target = tmp_path / "reference.json"
    source.write_text(json.dumps({"valid": True}))
    with pytest.raises(ValueError):
        promote_grasp_snapshot(source, target)
    assert not target.exists()
