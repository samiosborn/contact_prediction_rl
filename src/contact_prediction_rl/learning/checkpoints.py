# src/contact_prediction_rl/learning/checkpoints.py

"""Log directory and checkpoint resolution for PPO runs"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def latest_checkpoint_in_directory(
    directory: Path,
) -> Path:
    """Return the checkpoint with the highest iteration number"""

    checkpoints = list(
        directory.glob(
            "model_*.pt"
        )
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No model_*.pt checkpoints found in: {directory}"
        )

    def iteration(
        path: Path,
    ) -> int:
        try:
            return int(
                path.stem.split(
                    "_"
                )[-1]
            )

        except ValueError:
            return -1

    return max(
        checkpoints,
        key=iteration,
    )


def resolve_checkpoint(
    checkpoint: str | None,
    log_root_path: Path,
) -> Path | None:
    """Resolve a checkpoint argument"""

    if checkpoint is None:
        return None

    if checkpoint == "latest":
        run_directories = sorted(
            (
                path
                for path in log_root_path.iterdir()
                if path.is_dir()
            ),
            reverse=True,
        ) if log_root_path.exists() else []

        for run_directory in run_directories:
            try:
                return (
                    latest_checkpoint_in_directory(
                        run_directory
                    )
                )

            except FileNotFoundError:
                continue

        raise FileNotFoundError(
            f"No PPO checkpoints found under: {log_root_path}"
        )

    checkpoint_path = Path(
        checkpoint
    ).expanduser().resolve()

    if checkpoint_path.is_dir():
        return (
            latest_checkpoint_in_directory(
                checkpoint_path
            )
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    return checkpoint_path


def get_log_root_path(
    experiment_name: str,
) -> Path:
    """Return the directory holding all runs of an experiment"""

    return Path(
        os.path.abspath(
            os.path.join(
                "logs",
                "ppo",
                experiment_name,
            )
        )
    )


def create_log_directory(
    experiment_name: str,
    run_name: str,
) -> tuple[Path, Path]:
    """Create the root and run-specific logging directories"""

    log_root_path = get_log_root_path(
        experiment_name
    )

    print(
        f"[INFO] Logging experiment in directory: "
        f"{log_root_path}"
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    print(
        f"Exact experiment name requested from command line: "
        f"{timestamp}"
    )

    directory_name = timestamp

    if run_name:
        directory_name += (
            f"_{run_name}"
        )

    log_dir = (
        log_root_path
        / directory_name
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        log_root_path,
        log_dir,
    )
