# contact_prediction_rl

- Reinforcement-learning experiments for studying whether auxiliary contact prediction improves dexterous manipulation.
- The initial task uses a ROBOTIS HX5-D20 right hand in Isaac Lab to manipulate an object in simulation.
- The policy is trained with PPO while an auxiliary head predicts binary fingertip-object contact from the same learned representation.
- The main comparison is baseline PPO against PPO with auxiliary fingertip contact prediction.
- Contact is used as privileged supervision during training but is not provided as an input to the deployed policy.

## Environment

- The registered Isaac Lab environment is: ContactPredictionRl-Reorientation-HX5-D20-Right

- The simulated hand has 20 actuated joints.
- Five object-specific contact signals are defined for:
  - thumb
  - index
  - middle
  - ring
  - little
- These signals are derived from simulator contact forces at the corresponding fingertip links.

## Setup

### Install

git clone https://github.com/samiosborn/contact_prediction_rl.git
cd contact_prediction_rl

Optional Isaac Sim (large download, needed only for PhysX/Kit features): `uv sync --extra isaacsim`

## Run

- List the registered environments: `uv run python scripts/list_envs.py --show_presets`

- Train (headless is the default in Isaac Lab 3.0):
  `uv run python scripts/train.py --task ContactPredictionRl-Reorientation-HX5-D20-Right --num_envs 4096`

- Short check: add `--num_envs 64 --max_iterations 3`

- Play back a checkpoint:
  `uv run python scripts/play.py --task ContactPredictionRl-Reorientation-HX5-D20-Right --num_envs 16 --checkpoint logs/rsl_rl/<experiment>/<run>/model_<n>.pt`

- Run the HX5-D20 development test:
  `uv run python scripts/test_hx5.py --num_envs 1 --viz newton_gl physics=newton_mjwarp`

`scripts/train.py` and `scripts/play.py` import `contact_prediction_rl.tasks` (which registers the environments) and then call Isaac Lab's unified `isaaclab_rl.entrypoints` CLI, as Isaac Lab's own scripts do.

## Development

- Run tests without the simulator: `uv run pytest -m "not gpu"`

- Run the GPU smoke test (builds every registered task with 4 envs and steps it): `uv run pytest -m gpu`

- Run everything: `uv run pytest`

- Run formatting and code-quality hooks with: uv run pre-commit run --all-files

- Local editor configuration, virtual environments, logs, training outputs and generated caches are excluded from version control.
