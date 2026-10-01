# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
