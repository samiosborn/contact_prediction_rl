"""Promote a validated local grasp into versioned reference data, preserving bytes.

Run once in your repo checkout, then git add/commit the resulting JSON.
"""

from __future__ import annotations

import argparse

from contact_prediction_rl.tasks.reorientation.config.paths import GENERATED_GRASP_PATH, REFERENCE_GRASP_PATH
from contact_prediction_rl.tasks.reorientation.grasp_snapshot import promote_grasp_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default=str(GENERATED_GRASP_PATH))
    parser.add_argument("--destination", type=str, default=str(REFERENCE_GRASP_PATH))
    args = parser.parse_args()
    digest = promote_grasp_snapshot(args.source, args.destination)
    print(f"REFERENCE_GRASP_READY path={args.destination} sha256={digest}")
    print("Track this exact file in Git before the next training experiment.")


if __name__ == "__main__":
    main()
