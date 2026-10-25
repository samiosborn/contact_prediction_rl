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
