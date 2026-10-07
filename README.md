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

## Franka + HX5-D20 tasks

- Two tasks use a Franka Emika Panda arm carrying the HX5-D20 right hand (a single combined USD, see
  `src/contact_prediction_rl/assets/data/franka_hx5d20/SOURCE.md`):
  - `ContactPredictionRl-Reach-Franka-HX5-D20-Right`: move the hand to random poses with the fingers open
  - `ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right`: grasp a 6x6x12 cm block from a table and hold it 10 cm higher
- Both use PhysX, so install Isaac Sim once with `uv sync --extra isaacsim` and run with `uv run --extra isaacsim ...`.

- Train the lift task:
  `uv run --extra isaacsim python scripts/train.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right --num_envs 1024 --max_iterations 1000`

- Evaluate a checkpoint with deterministic rollouts (held at the goal at the end, block tilt, ...):
  `uv run --extra isaacsim python scripts/franka_hx5d20/evaluate_franka_hx5d20_lift.py --num_envs 512 --checkpoint logs/rsl_rl/lift_block_franka_hx5d20/<run>/model_<n>.pt`

- `scripts/franka_hx5d20/` also holds the two tools used to design the grasp: the arm pose solver
  (`find_franka_hx5d20_pregrasp.py`) and the table-grasp search (`search_franka_hx5d20_table_grasp.py`); each script
  documents its usage at the top.

## Development

- Run tests without the simulator: `uv run pytest -m "not gpu"`

- Run the GPU smoke test (builds every registered task with 4 envs and steps it): `uv run pytest -m gpu`

- Run everything: `uv run pytest`

- Run formatting and code-quality hooks with: uv run pre-commit run --all-files

- Local editor configuration, virtual environments, logs, training outputs and generated caches are excluded from version control.
