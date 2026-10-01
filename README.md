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

- This repository is an external Isaac Lab project using a standard `uv` `src` layout.

- The project currently uses a local Isaac Lab checkout through editable paths in `pyproject.toml`.

- The expected layout is:
~/Apps/IsaacLab
~/Git/Robotics/contact_prediction_rl

- If either repository moves, update the paths under `[tool.uv.sources]` in `pyproject.toml`.
- The default configuration uses the Newton backend and does not require Isaac Sim.

## Run

- List the registered environments: uv run python scripts/list_envs.py --show_presets

- Run the HX5-D20 development test:
uv run python scripts/test_hx5.py \
  --num_envs 1 \
  --viz newton_gl \
  physics=newton_mjwarp


## Project structure

contact_prediction_rl/
├── scripts/
│   ├── list_envs.py
│   └── test_hx5.py
├── src/
│   └── contact_prediction_rl/
│       ├── assets/
│       │   └── data/
│       │       └── hx5_d20/
│       │           ├── left/
│       │           └── right/
│       └── tasks/
│           └── reorientation/
│               ├── mdp/
│               └── config/
│                   └── hx5_d20_right/
├── tests/
├── pyproject.toml
└── uv.lock

- `tasks/reorientation/mdp` contains task-level MDP components.
- `tasks/reorientation/config/hx5_d20_right` contains the HX5-D20 scene, environment configuration, task registration and PPO configuration.
- Robot assets are stored under `assets/data/hx5_d20`.
- The right-hand model currently includes the generated URDF and USD used by Isaac Lab.

## Development

- Run tests with: uv run pytest

- Run formatting and code-quality hooks with: uv run pre-commit run --all-files

- Generate editor configuration with: uv run isaaclab --editor

- Local editor configuration, virtual environments, logs, training outputs and generated caches are excluded from version control.
