# Reorientation training gate

The diagnostic task keeps its existing 40D policy inputs and 5D contact labels.
Training is opt-in, since a grasp is not validated merely by compiling its configuration.

1. Run the revised feasibility diagnostic and inspect GRASP_SNAPSHOT_SAVED.
2. Export REORIENTATION_GRASP_SNAPSHOT to the absolute JSON path.
3. Validate the direct reset on your GPU:

    uv run python scripts/reorientation/validate_training_reset.py --num-envs 32 --repeats 5

4. Only once RESET_OK passes, smoke-test PPO:

    uv run python scripts/train.py --task ContactPredictionRl-Reorientation-Train-HX5-D20-Right --num_envs 16 --max_iterations 2 --seed 42 --no_random_episode_lengths --run_name reorientation_smoke agent.contact_prediction.loss_weight=0.0

The task uses a fixed +15-degree goal, 58D policy observations, 20D grasp-centred
actions, a measured finger equilibrium, and five separate contact labels.

GPU validation and task success are NOT established by these source changes.

## Read-only diagnostics

The following scripts leave the policy observations, rewards, contact labels,
and action terms unchanged. They disable success/drop **terminations only in the
diagnostic configuration** to inspect a continuous 150-step trajectory. Do not
interpret their reported success fractions as training episode success rates.

```bash
export REORIENTATION_GRASP_SNAPSHOT="$PWD/runs/reorientation/grasp_snapshot.json"

# Zero-action, Gaussian and untrained-policy trajectories:
uv run python scripts/reorientation/diagnose_training.py --mode zero
uv run python scripts/reorientation/diagnose_training.py --mode random --action-std 0.2
uv run python scripts/reorientation/diagnose_training.py --mode untrained

# Recreate the physically tested signed gait and one-way offset through the
# TRAINING task's live joint-position action conversion:
uv run python scripts/reorientation/replay_training_actions.py --controller gait --pattern spread --amplitude -0.6
uv run python scripts/reorientation/replay_training_actions.py --controller offset --pattern thumb_index --amplitude 0.6
```

The live action term offset and scale are inverted rather than assuming that a
policy action has the same numerical units as a joint target. Reported diagnostics
include angle, signed twist, cube drift, opposing contact, finger-mask diversity,
joint tracking and the fraction of physically clipped targets.
