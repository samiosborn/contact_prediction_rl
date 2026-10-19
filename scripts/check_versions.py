# scripts/check_versions.py

"""Print installed versions of the Isaac Lab stack and check them against the pins in pyproject.toml."""

import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.specifiers import SpecifierSet

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Check both exact and compatible-release constraints from uv overrides
CHECKED = ["torch", "torchvision", "torchaudio", "newton", "mujoco", "mujoco-warp"]
REPORTED = ["isaaclab", "isaacsim", "rsl-rl-lib"]


def expected_pins() -> dict[str, tuple[str, str]]:
    """Return operators and versions from uv overrides, keyed by distribution name."""
    data = tomllib.loads(PYPROJECT.read_text())
    pins = {}
    for req in data["tool"]["uv"]["override-dependencies"]:
        m = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?\s*(==|~=)\s*([\w.]+)", req)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = (m.group(2), m.group(3))
    return pins


def installed(dist: str) -> str | None:
    try:
        return metadata.version(dist).split("+")[0]
    except metadata.PackageNotFoundError:
        return None


def mismatches() -> list[str]:
    """Return human-readable mismatches between installed versions and pyproject pins."""
    pins = expected_pins()
    problems = []
    for dist in CHECKED:
        op, want = pins.get(dist, (None, None))
        have = installed(dist)
        if want is None:
            continue
        if have is None:
            problems.append(f"{dist}: not installed (expected {op}{want})")
        elif not SpecifierSet(f"{op}{want}").contains(have):
            problems.append(f"{dist}: installed {have}, pyproject expects {op}{want}")
    # Isaac Lab is pinned in project dependencies rather than uv overrides
    for req in tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]:
        m = re.match(r"^isaaclab(?:\[[^\]]*\])?==([\w.]+)$", req)
        if m and installed("isaaclab") != m.group(1):
            problems.append(f"isaaclab: installed {installed('isaaclab')}, pyproject pins =={m.group(1)}")
    return problems


def main() -> int:
    print(f"python     {sys.version.split()[0]}")
    for dist in CHECKED + REPORTED:
        print(f"{dist:16s} {installed(dist) or '(not installed)'}")
    try:
        import torch

        print(f"cuda       {torch.version.cuda} (available: {torch.cuda.is_available()})")
    except ImportError:
        pass
    problems = mismatches()
    for p in problems:
        print("MISMATCH:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
