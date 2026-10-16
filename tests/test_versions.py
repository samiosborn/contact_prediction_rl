# tests/test_versions.py

"""Installed dependency versions must match the pins declared in pyproject.toml."""

import importlib.util
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
import check_versions  # noqa: E402


@pytest.mark.unit
def test_versions_match_pyproject():
    assert check_versions.mismatches() == []


@pytest.mark.unit
def test_isaaclab_is_pinned_release():
    assert check_versions.installed("isaaclab") is not None
    assert importlib.util.find_spec("isaaclab_tasks") is not None


@pytest.mark.unit
@pytest.mark.parametrize('version, accepted', [('3.10.9', False), ('3.11.0', True), ('3.11.1', True), ('3.12.0', False), ('3.110.0', False)])
def test_compatible_release_bounds(monkeypatch, version, accepted):
    """Compatible releases must satisfy both the lower bound and release prefix."""
    original = check_versions.installed
    monkeypatch.setattr(check_versions, 'installed', lambda dist: version if dist == 'mujoco' else original(dist))
    problems = check_versions.mismatches()
    assert (not any(problem.startswith('mujoco:') for problem in problems)) == accepted
