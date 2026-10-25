"""Validated Reorientation reset state, exported by the Newton feasibility runner."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def load_grasp_snapshot(path: str | Path) -> dict:
    """Reject incomplete or malformed snapshots instead of training from an unverified pose."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Validated grasp snapshot not found: {path}. "
            "Promote your local runs/reorientation/grasp_snapshot.json with "
            "uv run python scripts/reorientation/promote_grasp_snapshot.py"
        )
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or not snapshot.get("valid", False):
        raise ValueError("Reorientation requires a validated version-1 grasp snapshot")
    for group, expected in (
        ("arm_joint_pos", {f"panda_joint{i}" for i in range(1, 8)}),
        ("finger_joint_pos", {f"finger_r_joint{i}" for i in range(1, 21)}),
        ("finger_target_pos", {f"finger_r_joint{i}" for i in range(1, 21)}),
    ):
        values = snapshot.get(group, {})
        if set(values) != expected or not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Incomplete or non-finite grasp snapshot: {group}")
    for name, size in (("cube_pos_w", 3), ("cube_quat_w", 4), ("axis_world", 3)):
        values = snapshot.get(name)
        if not isinstance(values, list) or len(values) != size or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid grasp snapshot: {name}")
    if abs(sum(v * v for v in snapshot["cube_quat_w"]) - 1.0) > 0.01:
        raise ValueError("Cube quaternion must be unit length (XYZW)")
    if abs(sum(v * v for v in snapshot["axis_world"]) - 1.0) > 0.01:
        raise ValueError("Palm rotation axis must be unit length")
    return snapshot



def grasp_snapshot_sha256(path: str | Path) -> str:
    """Hash the exact input bytes, including formatting, for experiment provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def promote_grasp_snapshot(source: str | Path, destination: str | Path) -> str:
    """Promote validated bytes into a versioned canonical reference without rewriting them."""
    source, destination = Path(source), Path(destination)
    load_grasp_snapshot(source)
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite a different canonical grasp: {destination}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return digest
