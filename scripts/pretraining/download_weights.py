# scripts/pretraining/download_weights.py

"""Download pretrained policy weights into the Hugging Face cache (outside the repository; weights are not redistributed).

On a cluster, run this on a login node with internet access and point HF_HOME at shared storage.
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from contact_prediction_rl.pretraining.models import LEROBOT_POLICIES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=list(LEROBOT_POLICIES), help="Models to download.")
    parser.add_argument("--cache_dir", default=None, help="Defaults to the Hugging Face cache (HF_HOME).")
    args = parser.parse_args()

    for name in args.models:
        repo_id = LEROBOT_POLICIES[name][1]
        path = snapshot_download(repo_id, cache_dir=args.cache_dir)
        print(f"{name}: {repo_id} -> {path}")


if __name__ == "__main__":
    main()
