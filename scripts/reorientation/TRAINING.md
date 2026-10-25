# Reorientation training gate

The training task starts every episode from the versioned, validated physical
grasp at `data/reorientation/grasps/hx5_d20_palm_up_v1.json`.
This reference is selected by `tasks/reorientation/config/paths.py` and
`ReorientationTrainEnvCfg.grasp_snapshot_path`, not by an environment variable.

**One-time local migration (required before running the changed task):**

```bash
uv run python scripts/reorientation/promote_grasp_snapshot.py
git add data/reorientation/grasps/hx5_d20_palm_up_v1.json
git commit -m "Add validated HX5 palm-up grasp reference"
git push origin feature/reorientation-feasibility
```

The utility validates and copies the exact bytes from the old
`runs/reorientation/grasp_snapshot.json`. It will not replace a different
canonical snapshot. A missing reference fails fast with a promotion hint.
The input's SHA-256 is included in the environment config saved to the run's
`params/env.yaml`; baseline and auxiliary runs should have matching hashes.

**Physics gate (after promotion):**

```bash
uv run python scripts/reorientation/validate_training_reset.py --num-envs 32 --repeats 5
uv run python scripts/reorientation/diagnose_training.py --mode zero
uv run python scripts/reorientation/replay_training_actions.py --controller gait --pattern spread --amplitude -0.6
```

These diagnostic scripts disable success and drop terminations **locally** to
measure continuous trajectories; they do not modify PPO or training rewards.
The feasibility experiment still exports *candidate* grasps into `runs/`.
Never promote a new candidate over v1 without explicitly versioning it.

The 58D policy observation, 20D action space, five contact labels, reward
and termination logic are unaffected by this data migration.

**Check unexpected fast PPO success (300-iteration baseline):**

```bash
uv run python scripts/reorientation/diagnose_training.py --mode trained --checkpoint latest
```

This uses deterministic policy actions and records goal error at the first two
steps and beyond with success termination disabled. Do not interpret near-100%
training success at two steps as proven sustained reorientation without this
independent physics audit.


## Fixed-horizon task (updated)

Reorientation Train terminates **only at the six-second time limit**, about
180 control steps with the existing 120 Hz physics / four-step decimation.
Neither briefly reaching the target nor dropping the cube ends the episode.

New reward terms in `mdp/rewards.py`:

- `orientation` (weight +1): bounded progress toward the +15° target,
  gated on thumb/opposing-finger grasp retention
- `stable_hold` (weight +4): one for each step within 5° and 20 mm,
  with opposing contact, linear speed ≤0.10 m/s and angular speed ≤0.50 rad/s
- `dropped` (weight −2): each step after the cube drifts beyond 40 mm

Isaac Lab reward weights are scaled by the control timestep. Hold speed
thresholds and relative weights are **initial hypotheses**, not calibrated
against complete Newton reference trajectories. In particular, do not
infer stable manipulation from the original 300-iteration quick-flick baseline.

The `stable_hold` reward term reuses pure predicates in
`tasks/reorientation/metrics.py`. Its independent per-env tracker in
`diagnostics.py` reports episode summaries through the existing Isaac Lab
`extras["log"]` path on time-limit reset. The current PPO runner already
writes those scalar values to console and TensorBoard as `Metrics/*`:
`sustained_success_rate`, `stable_goal_fraction`, `transient_goal_rate`,
`grasp_retention_fraction`, `dropped_fraction`, final and minimum goal
errors, max displacement and final stable-goal rate.

W&B, changes to `scripts/train.py`, a new
`scripts/reorientation/evaluate_training.py` and action limits are postponed.
