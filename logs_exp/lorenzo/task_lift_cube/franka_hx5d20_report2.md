# Lift task: checking the new actor architecture (branch `sami`)

## Summary
Sami's branch replaces RSL-RL's default `MLPModel` actor with an explicit `Actor` (observation encoder, latent z, action head) and moves all RSL-RL hyperparameters into `learning/configs/rsl_rl.py`. Before merging, we retrained the lift task to check that the new setup matches the earlier results.

- The first run with the new `Actor` (seed 42) **never learned to grasp**. Other seeds do learn.
- The new `Actor` is **functionally identical to `MLPModel`**. The failure was seed luck, not an architecture bug.
- The lift task is **very sensitive to the seed**, so single runs are not enough to compare configurations.

All runs: task `ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right`, 1024 envs, 300 iterations (the first run went on to about 600).

## 1. Results

| Run | Actor | Seed | Success at it. 299 | Held at goal | Mean reward | Outcome |
|---|---|---|---|---|---|---|
| `2026-10-07_11-21-56` | `MLPModel` | 42 | 1.00 | 0.84 | 175 | Learns (earlier reference run) |
| `2026-10-08_16-25-40` | `Actor` | 42 | 0 (still 0 at it. 600) | 0 | 6.8 | Never grasps |
| `2026-10-08_16-42-22` | `MLPModel` (`agent.actor.class_name=MLPModel`) | 42 | 1.00 | 0.84 | 175 | Identical to the reference run |
| `2026-10-08_16-53-20` | `Actor` | 1 | 0.91 | 0.79 | 161 | Learns, slowly (starts lifting after it. 75) |
| `2026-10-08_17-04-26` | `Actor` | 2 | 0.30 | 0 | 46 | Learns early (success 0.80 at it. 75), then degrades |

The curves were sampled at 5 points per run, so finer wiggles are not captured here.

- **Seed 42 with `Actor`:** the policy reaches toward the block (`reaching_object`, `lift_progress` rise) but `hand_closed_near_object` decays to about 0 and `lifting_object` stays 0. The action std collapsed from 0.2 to 0.058, the entropy fell from -1.5 to -12, and the adaptive learning rate dropped to 1.5e-5. With no exploration left, the policy could not recover.
- **Seed 42 with `MLPModel` reproduces the earlier run exactly**, metric for metric. So the config refactor is correct, and training is deterministic for a given seed.

## 2. Is the new `Actor` different from `MLPModel`?

No functional difference was found. Checked against the RSL-RL source and by building both models with the lift settings (hidden dims [256, 128, 64], ELU, observation normalisation, Gaussian init std 0.2):

- Same layers: Linear 40-256, ELU, 128, ELU, 64, ELU, then a Linear to 7 outputs. Both have 52,110 parameters.
- After copying the `MLPModel` weights into the `Actor`, the outputs match (`allclose`).
- The std parameter is 0.2 in both. `GaussianDistribution.init_mlp_weights` is a no-op, so the `Actor` calling it changes nothing.
- The initial weight spread is the same.

**The one difference is the random initialisation.** `Actor.__init__` calls `super().__init__()`, which builds a default MLP, then discards it and builds a second one. The throwaway MLP uses up random numbers, so for a given seed the `Actor` starts from different weights than `MLPModel` does. This explains why seed 42 behaves differently between the two classes. It is not a flaw in the architecture.

## 3. Why the seeds differ so much

This is my interpretation of the curves. Only the first point is well supported, and none of these was tested separately.

1. **Closing the hand on the block is a rare event at the start.** The lift reward only pays after a grasp, but early actions are random (std 0.2). Which early grasps happen to work depends on the seed. Seed 1 needed about 150 iterations to find one. Seed 42 with `Actor` never did.
2. **A learned grasp can be lost.** Seed 2 found the grasp by iteration 75, then position error rose from 0.09 to 0.18 and reward fell from 119 to 46. The `MLPModel` run also dipped (success 0.82, 0.42, then 1.0).
3. **The adaptive learning rate swings a lot** (1e-5 to 1.7e-4 in these logs), which makes collapses and recoveries look random across seeds.
4. **The seed also changes** the block and goal sampling and the action noise, so early experience differs from the start.

## 4. Consequences and next steps

- The `sami` branch was merged: the `Actor` can learn the task, and the refactored config reproduces the earlier run with `MLPModel`.
- **One seed says very little.** For the baseline vs. auxiliary contact loss comparison, use at least 3-5 seeds per configuration and compare means and spreads.
- Compare final policies with `evaluate_franka_hx5d20_lift.py` (deterministic rollouts), not the noisy training curves. Also check intermediate checkpoints, since the last one is not always the best (seed 2).
- Possible ways to steady training, none tried yet: cap the learning rate below what the adaptive schedule reaches, raise `desired_kl` or lower `num_learning_epochs` so each update changes the policy less, and train for more iterations.
- Optional clean-up: avoid building the MLP twice in `Actor.__init__`. This changes which weights each seed gives, so earlier seed results would not carry over.
