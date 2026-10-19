# contact_prediction_rl

## Research Aim

- Study whether auxiliary contact prediction improves representation learning and control in dexterous manipulation.
- The policy is trained with PPO while an auxiliary head predicts binary fingertip-object contact from the same learned representation.
- The main comparison is baseline PPO against PPO with auxiliary fingertip-contact prediction.
- Contact is privileged supervision during training and is not provided to the deployed policy.

## Policy Architecture

The actor separates observation encoding from action prediction:

```text
task observation
    ↓
ObservationEncoder
    ↓
latent representation z
    ↓
action head
    ↓
Gaussian policy
```

The encoder architecture is shared across tasks, while the input and output dimensions are inferred from each task's observation and action spaces.

PPO is implemented locally in `src/contact_prediction_rl/learning/`. Shared architecture and PPO settings are defined in:

```text
src/contact_prediction_rl/learning/configs/ppo.py
```

Each task builds its own PPO configuration from these shared settings in `agents/ppo_cfg.py`, registered under `ppo_cfg_entry_point`.

The actor implementation is defined in:

```text
src/contact_prediction_rl/learning/models/actor.py
```

The critic is a separate MLP (`learning/models/critic.py`).

## Manipulator

The primary manipulator is the ROBOTIS HX5-D20 right hand mounted on a Franka Emika Panda arm.

- 20 actuated hand joints
- Five fingertip contact signals:
  - thumb
  - index
  - middle
  - ring
  - little
- Contact labels are derived from simulator contact forces at the corresponding fingertip links.

## Setup

Clone the repository and install the default dependencies:

```bash
git clone https://github.com/samiosborn/contact_prediction_rl.git
cd contact_prediction_rl
uv sync
```

### Physics and robot assets

The Franka + HX5-D20 tasks default to Newton MJWarp. PhysX is also supported, while the `kit` visualiser requires Isaac Sim.

The combined Franka + HX5-D20 USD is stored with Git LFS:

```bash
sudo apt install git-lfs
git lfs install
git lfs pull
```

The training, playback, and evaluation entrypoints accept the NVIDIA Omniverse EULA non-interactively by default. An existing `OMNI_KIT_ACCEPT_EULA` value is preserved. For GPU tests, set `OMNI_KIT_ACCEPT_EULA=YES` as shown below.

## Validation

List the registered environments:

```bash
uv run python scripts/list_envs.py --show_presets
```

Run tests that do not launch the simulator:

```bash
uv run pytest -m "not gpu"
```

Test the actor architecture and gradient flow through the observation encoder:

```bash
uv run pytest tests/test_actor.py -v
```

Run GPU smoke tests for all registered tasks:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run pytest -m gpu
```

Run the complete test suite:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run pytest
```

## Tasks

### Franka + HX5-D20 Reorientation

`ContactPredictionRl-Reorientation-HX5-D20-Right`

An experimental environment for dexterous manipulation and fingertip-contact research. A meaningful object-reorientation objective has not yet been implemented.

Five binary fingertip-object contact labels are available. The existing labels use force-thresholded distal-link contact, while a separate diagnostic classifies Newton contact points against candidate fingertip patches.

Validate all five fingers:

```bash
uv run python scripts/validate_fingertip_contacts.py \
  --finger all \
  --cube-mode fixed \
  --measure-patches \
  --visualizer newton_gl
```

Inspect the candidate patches:

```bash
uv run python scripts/validate_fingertip_contacts.py \
  --inspect-only \
  --show-patches \
  --visualizer newton_gl
```

The five isolated contact trials and a proximal outside-patch negative control have been validated using Newton MJWarp. The patches are geometric proxies, not calibrated tactile sensors.

### Franka + HX5-D20 Reach

`ContactPredictionRl-Reach-Franka-HX5-D20-Right`

Move the Franka-mounted HX5-D20 hand to random end-effector poses with the fingers open.

Train:

```bash
uv run python scripts/train.py \
  --task ContactPredictionRl-Reach-Franka-HX5-D20-Right
```

Checkpoints are written to `logs/ppo/<experiment>/<run>/model_<n>.pt`.

### Franka + HX5-D20 Lift

`ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right`

Grasp a 6 × 6 × 12 cm block from a table and hold it 10 cm above its initial position.

Train:

```bash
uv run python scripts/train.py \
  --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right \
  --num_envs 1024 \
  --max_iterations 1000
```

Run a short training check:

```bash
uv run python scripts/train.py \
  --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right \
  --num_envs 64 \
  --max_iterations 3
```

Evaluate a checkpoint with deterministic rollouts (`--checkpoint` accepts a file, a run directory or `latest`):

```bash
uv run python scripts/franka_hx5d20/evaluate_franka_hx5d20_lift.py \
  --num_envs 512 \
  --checkpoint logs/ppo/lift_block_franka_hx5d20/<run>/model_<n>.pt
```

Watch a trained policy in the viewport, or record a video:

```bash
uv run python scripts/play.py \
  --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right \
  --checkpoint latest --visualizer kit --real_time

uv run python scripts/play.py \
  --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right \
  --checkpoint latest --video
```

Videos are written to `<run directory>/videos/play/` and need the `video` extra (MoviePy). `scripts/train.py --video`
records training clips to `<run directory>/videos/train/` the same way.

### Franka + HX5-D20 Lift with hand synergies

`ContactPredictionRl-Lift-Block-Franka-HX5-D20-Synergy-Right`

The same block-lift task, but the policy blends three hand-designed finger shapes (7 arm + 3 hand actions) instead of
one open/close value:

| Action | Shape |
|---|---|
| curl | bending joints of index, middle, ring and little |
| thumb | thumb opposition and flexion |
| spread | finger abduction, fanning the fingers apart |

Each synergy has a closure in [0, 1] that moves at the same limited rate as the single-closure hand; a negative action
closes it and a non-negative action opens it. The shapes are defined in
`src/contact_prediction_rl/robots/franka_hx5d20.py` (`FRANKA_HX5D20_HAND_SYNERGIES`). Train, evaluate and play work as
for the lift task, with the new task name; checkpoints go to `logs/ppo/lift_block_franka_hx5d20_synergy/`. The
evaluation script also reports the final closure of each synergy.

Scripted grasps over the synergy space (no training) can be run with:

```bash
uv run python scripts/franka_hx5d20/search_franka_hx5d20_table_grasp.py \
  --gravities "0,-1,0" --centers "0.04,0,0.09" --block_height 0.12 --approach 0.06 \
  --synergies "1,1,0;1,0.5,0.5;1,0,0"
```

### Franka + HX5-D20 Lift with every finger joint controlled

`ContactPredictionRl-Lift-Block-Franka-HX5-D20-AllJoints-Right`

The same block-lift task, but the policy controls each of the 20 finger joints directly (7 arm + 20 hand actions)
with no designed finger posture. Each hand action moves one joint target at a limited rate: the whole joint range in
2 s, the same speed limit as the single-closure hand. An action of 0 holds the posture and a positive action raises the
joint angle. The hand starts open, and the closed-hand reward counts how far the targets are along the curl posture.
Checkpoints go to `logs/ppo/lift_block_franka_hx5d20_all_joints/`, and the evaluation script also prints the final
target of every hand joint.

```bash
uv run --extra isaacsim python scripts/train.py \
  --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-AllJoints-Right \
  --num_envs 1024 --max_iterations 600 --seed 42
```

The combined Franka + HX5-D20 asset is documented in:

```text
src/contact_prediction_rl/assets/data/franka_hx5d20/SOURCE.md
```

Additional grasp-design tools are in `scripts/franka_hx5d20/`:

- `find_franka_hx5d20_pregrasp.py` — arm pose solver
- `search_franka_hx5d20_table_grasp.py` — scripted table-grasp search

## Development

`scripts/train.py` and `scripts/play.py` import `contact_prediction_rl.tasks` to register the environments, then resolve the environment and PPO configurations from the Gym registry (`env_cfg_entry_point`, `ppo_cfg_entry_point`). PPO hyperparameters can be overridden on the command line with Hydra syntax, for example `agent.algorithm.learning_rate=1e-4`.

Run formatting and code-quality hooks:

```bash
uv run pre-commit run --all-files
```

Local editor configuration, virtual environments, logs, training outputs, generated caches, and other machine-specific files are excluded from version control.
