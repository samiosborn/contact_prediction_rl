# contact_prediction_rl

## Research Aim

- Study whether auxiliary contact prediction improves representation learning and control in dexterous manipulation.
- The policy is trained with PPO while an auxiliary head predicts binary fingertip-object contact from the same learned representation.
- The main comparison is baseline PPO against PPO with auxiliary fingertip-contact prediction.
- Contact is privileged supervision during training and is not provided to the deployed policy.

## Manipulator

The primary manipulator is the ROBOTIS HX5-D20 right hand.

- 20 actuated joints
- Five fingertip contact signals:
  - thumb
  - index
  - middle
  - ring
  - little
- Contact labels are derived from simulator contact forces at the corresponding fingertip links.

The hand is used either standalone for reorientation or mounted on a Franka Emika Panda arm for reach and lift tasks.

## Setup

Clone the repository and install the default dependencies:

```bash
git clone https://github.com/samiosborn/contact_prediction_rl.git
cd contact_prediction_rl
uv sync
```

### PhysX / Isaac Sim

The Franka + HX5-D20 tasks use PhysX and require Isaac Sim.

The combined Franka + HX5-D20 USD is stored with Git LFS:

```bash
sudo apt install git-lfs
git lfs install
git lfs pull
```

Install the Isaac Sim dependencies:

```bash
uv sync --extra isaacsim
```

Isaac Sim requires acceptance of the NVIDIA Omniverse EULA. To launch it non-interactively:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

## Validation

List the registered environments:

```bash
uv run python scripts/list_envs.py --show_presets
```

Run tests that do not launch the simulator:

```bash
uv run pytest -m "not gpu"
```

Run GPU smoke tests for all registered tasks:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim pytest -m gpu
```

Run the complete test suite:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim pytest
```

## Tasks

### HX5-D20 Reorientation

`ContactPredictionRl-Reorientation-HX5-D20-Right`

Standalone HX5-D20 in-hand reorientation using the Newton backend.

Train:

```bash
uv run python scripts/train.py \
  --task ContactPredictionRl-Reorientation-HX5-D20-Right \
  --num_envs 4096
```

Short training check:

```bash
uv run python scripts/train.py \
  --task ContactPredictionRl-Reorientation-HX5-D20-Right \
  --num_envs 64 \
  --max_iterations 3
```

Play a checkpoint:

```bash
uv run python scripts/play.py \
  --task ContactPredictionRl-Reorientation-HX5-D20-Right \
  --num_envs 16 \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_<n>.pt
```

Run the standalone hand development test:

```bash
uv run python scripts/test_hx5.py \
  --num_envs 1 \
  --viz newton_gl \
  physics=newton_mjwarp
```

### Franka + HX5-D20 Reach

`ContactPredictionRl-Reach-Franka-HX5-D20-Right`

Move the Franka-mounted HX5-D20 hand to random end-effector poses with the fingers open.

Train:

```bash
uv run --extra isaacsim python scripts/train.py \
  --task ContactPredictionRl-Reach-Franka-HX5-D20-Right
```

### Franka + HX5-D20 Lift

`ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right`

Grasp a 6 × 6 × 12 cm block from a table and hold it 10 cm above its initial position.

Train:

```bash
uv run --extra isaacsim python scripts/train.py \
  --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right \
  --num_envs 1024 \
  --max_iterations 1000
```

Evaluate a checkpoint with deterministic rollouts:

```bash
uv run --extra isaacsim python scripts/franka_hx5d20/evaluate_franka_hx5d20_lift.py \
  --num_envs 512 \
  --checkpoint logs/rsl_rl/lift_block_franka_hx5d20/<run>/model_<n>.pt
```

The combined Franka + HX5-D20 asset is documented in:

```text
src/contact_prediction_rl/assets/data/franka_hx5d20/SOURCE.md
```

Additional grasp-design tools are in `scripts/franka_hx5d20/`:

- `find_franka_hx5d20_pregrasp.py` — arm pose solver
- `search_franka_hx5d20_table_grasp.py` — scripted table-grasp search

## Development

`scripts/train.py` and `scripts/play.py` import `contact_prediction_rl.tasks` to register the environments before calling Isaac Lab's reinforcement-learning entry points.

Run formatting and code-quality hooks:

```bash
uv run pre-commit run --all-files
```

Local editor configuration, virtual environments, logs, training outputs, generated caches, and other machine-specific files are excluded from version control.