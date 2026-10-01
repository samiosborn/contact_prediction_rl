# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training entry point for contact_prediction_rl tasks.

Delegates to Isaac Lab's unified train CLI after registering this project's tasks, e.g.

    python scripts/train.py --task ContactPredictionRl-Reorientation-HX5-D20-Right --rl_library rsl_rl
"""

# Warp captures ``enable_backward`` at module creation (import time), so set it before importing
# anything that defines Warp kernels. Mirrors Isaac Lab's own scripts/reinforcement_learning/train.py.
import warp as wp

wp.config.enable_backward = False

import contact_prediction_rl.tasks  # noqa: E402, F401  (registers the gym environments)
from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning library."""
    return run_train_cli(argv)


if __name__ == "__main__":
    # ``record`` surfaces the failing rank's traceback under torchrun; it is a no-op otherwise.
    from torch.distributed.elastic.multiprocessing.errors import record

    raise SystemExit(record(main)())
