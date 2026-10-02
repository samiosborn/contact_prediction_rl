# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback entry point for contact_prediction_rl tasks.

Delegates to Isaac Lab's unified play CLI after registering this project's tasks, e.g.

    python scripts/play.py --task ContactPredictionRl-Reorientation-HX5-D20-Right --rl_library rsl_rl
"""

# Warp captures ``enable_backward`` at module creation (import time), so set it before importing
# anything that defines Warp kernels. Mirrors Isaac Lab's own scripts/reinforcement_learning/play.py.
import warp as wp

wp.config.enable_backward = False

import contact_prediction_rl.tasks  # noqa: E402, F401  (registers the gym environments)
from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning library."""
    return run_play_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
