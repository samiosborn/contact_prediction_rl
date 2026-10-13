# Franka Panda + Robotis HX5-D20 hand in Isaac Lab: lifting a block with RL

## Summary
We built a new robot in Isaac Lab (develop branch): a Franka Emika Panda arm with a Robotis HX5-D20 five-finger right hand. We then trained it with reinforcement learning (RL) on two tasks:

1. **Reach**: move the hand to random target poses. Solved: 100 % success.
2. ==**Lift**: pick a block up from a flat table, lift it 10 cm and hold it there==. Solved: the best checkpoint holds the block at the target at the end of **99.4 %** of 512 test episodes, never drops it, and places it within about 1 mm of the target.

The lift task took the most work. Most of the difficulties were not in the RL algorithm itself: they were in the physics of grasping with this particular hand, and in rewards that the policy could exploit. Each problem was found by measuring, usually with a small scripted test that needs no training, before changing anything.

What is still not satisfying: the hand holds the block tilted by about 40°, pinched by the fingertips, instead of upright in the palm. The finger and thumb shape is not learned (see [What the policy learns](#what-the-policy-learns-and-what-it-does-not)), so improving it means either better placement of the hand or giving the policy control over the fingers (see [Next steps](#next-steps)).

---

## 1. The robot

### Building the combined robot

- The hand was converted from MuJoCo (MJCF) to USD with Isaac Lab's converter and attached to the Franka's `panda_hand` with a fixed joint. The combined robot lives in `source/isaaclab_assets/data/Robots/Robotis/FrankaHX5D20/franka_hx5d20.usd`.
- Two problems in the combined USD had to be fixed before anything could move:
	  - It had **no default prim**, so Isaac Lab could not reference it as a robot.
	  - It had **two articulation roots** (the arm and the hand base). Physics treats that as two robots glued together, and Isaac Lab refuses to spawn it. The hand base also carried a Newton-specific root schema that silently re-added the root, which had to be removed as well.
- A joint check script then drove all 27 joints (7 arm + 20 hand) one at a time: all moved, and the hand stayed rigidly attached to the arm (0.0 mm drift).

### Robot configuration (`FRANKA_HX5D20_CFG`)

| Part | Joints | Control | Stiffness | Damping | Max torque |
|---|---|---|---|---|---|
| Arm, shoulder | `panda_joint1-4` | position (implicit PD) | 400 | 80 | 87 N·m |
| Arm, forearm | `panda_joint5-7` | position (implicit PD) | 400 | 80 | 12 N·m |
| Hand | `finger_r_joint1-20` | position (implicit PD) | 20 | 2 | 10 N·m |

The hand has 4 thumb joints and 4 joints per finger (spread, then three bending joints).

---

## 2. Task 1: Reach

==The policy moves the hand to random target poses in front of the robot==. The fingers stay open.

- Built on Isaac Lab's Franka reach task (`Isaac-Reach-Franka-HX5D20`).
- **Actions**: 7 arm joint position targets. **Observations**: arm joint positions and velocities, the target pose, the last action.
- **Result**: after 1000 iterations, 100 % of targets reached (within 5 cm and 0.2 rad). Positioning error about 4.6 cm and 0.19 rad, both inside the success thresholds.

This proved the robot, its configuration and the training pipeline before moving to grasping.

---

## 3. Task 2: Lift a block from the table

### 3.1 The final task

| Item                  | Value                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------- |
| Task ID               | `Isaac-Lift-Cube-Franka-HX5D20`                                                         |
| Object                | Block, 6 × 6 × 12 cm, 0.1 kg, friction 1.0, standing on a flat table                    |
| Spawn                 | 45 cm in front of the robot, randomised ±1 cm                                           |
| Goal                  | ==**Fixed**: straight above the start, 10 cm higher, held until the end==               |
| Episode               | 8 s                                                                                     |
| Physics               | PhysX, 100 Hz simulation, control at 50 Hz (each action is applied for 2 physics steps) |
| Parallel environments | 1024 on one laptop GPU (RTX 5080), about 33,000 steps per second                        |
| Start                 | Open hand beside the block, palm facing it, 6 cm away                                   |

**Grasp type: ==a side grasp==.** The fingers point horizontally, the palm faces the block and the thumb points up. The hand approaches the block's side, not its top.

### 3.2 What the policy sees (observations)

72 numbers per step, all in the robot's own frame:

| Observation          | Size | Content                                                   |
| -------------------- | ---- | --------------------------------------------------------- |
| **==Joint positions==**  | 27   | All arm and hand joints, relative to their default values |
| **==Joint velocities==** | 27   | All arm and hand joints                                   |
| **==Block position==**   | 3    | Block centre, relative to the robot base                  |
| **==Goal==**             | 7    | Target position and orientation                           |
| **==Last action==**      | 8    | What the policy did at the previous step                  |

The observations are normalised (running mean and standard deviation) before entering the networks.

### 3.3 What the policy does (actions)

**8 numbers per step:**

- **7 arm values**: target angle of each arm joint = start angle + 0.2 × action.
- **1 hand value**: a binary command. Negative means close, otherwise open.

The hand value does **not** set the finger joints directly. It drives a single *closure fraction* from 0 (open) to 1 (closed) at a fixed rate (a full close takes 2 s), and all 20 finger joints follow:

```
finger targets = OPEN posture + closure × (CLOSED posture − OPEN posture)
```

The open posture is all finger joints at 0. The closed posture (thumb opposed, fingers bent to 75 % of their range) was designed with forward kinematics and scripted grasp tests.

### 3.4 What the policy learns, and what it does not

==**Not learned:** the shape of the hand. The 20 finger joints always follow the same fixed path between the two postures.== The fingers still adapt passively, because the closed posture is only a target: a finger stops when it touches the block and presses with the force its motor allows. Their final shape therefore follows the object through physics, not through learning.

<mark style="background: #FFB8EBA6; font-weight: bold;">Learned: everything else:</mark>

1. how to move the arm to put the hand around the block;
2. when to close the hand;
3. how to lift the block and bring it to the goal (a coordinated 7-joint motion);
4. how to hold it steady there;
5. all of this as feedback from the current state, so it adapts to where the block actually is.

In robotics terms, the hand is used like a gripper with one degree of freedom, with a hand-designed closing motion (a "synergy"). This shrinks the action space from 27 values to 8, which is why training takes minutes. The cost is that how natural the grasp looks depends on the designed posture and on where the policy puts the hand, and the policy controls only the second.

### 3.5 Rewards

Rewards are added up every step. <mark style="background: #ADCCFFA6;">"In hand" means the held part of the block is within 8 cm of the grasp centre of the hand</mark>. "Held part" is the upper 6 cm of the block, where the hand holds it.

| Reward | Weight | What it pays for |
|---|---|---|
| Reaching | 0.5 | Grasp centre of the hand close to the held part (length scale 3 cm) |
| Hand closed near the block | 4 | Closure fraction × closeness to the held part (3 cm). Pays only for a hand that is actually closed, and only near the block |
| Lift progress | 5 | Height of the block above rest, from 0 (on the table) to 1 (at the goal), only while in hand |
| Lifted | 5 | Block more than 2 cm above rest, only while in hand |
| Goal tracking (broad) | 8 | Block close to the goal, length scale 30 cm, only while lifted in hand |
| Goal tracking | 16 | Same, length scale 10 cm |
| Goal tracking (fine) | 5 | Same, length scale 2.5 cm |
| Hand orientation | 0.5 | Hand kept in the side-grasp orientation, only near the block |
| Action rate | −0.0001 | Penalty on changes between consecutive actions |
| Joint velocity | −0.0001 | Penalty on fast arm motion |

**Episode end:** time-out after 8 s, or the block falls off the table.

**Logged metrics:**

- `held_at_goal_at_end`: block held within 5 cm of the goal at the last step. This is the real success measure.
- `success_rate`: block passed within 5 cm of the goal at least once. Looser: it can be high even when the block does not stay at the goal.

### 3.6 Algorithm: PPO

<mark style="background: #FFB86CA6;">Training uses PPO (Proximal Policy Optimization)</mark>, Isaac Lab's default for this kind of task. PPO alternates between two phases:

1. collect experience from all environments with the current policy;
2. improve the policy with small, clipped steps, so that one update cannot change it too much.

| Setting                             | Value                                          | Meaning                                                             |
| ----------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------- |
| Networks                            | Actor and critic, 3 layers (256, 128, 64), ELU | The actor chooses actions; the critic estimates how good a state is |
| Steps per environment per iteration | 24                                             | 1024 × 24 ≈ 24,600 steps per update                                 |
| Learning rate                       | 1e-4, adaptive (target KL 0.01)                | Step size, automatically reduced when the policy changes too fast   |
| Discount factor γ                   | 0.99                                           | How far ahead rewards count: about 100 steps (2 s)                  |
| GAE λ                               | 0.95                                           | Smoothing of the advantage estimates                                |
| Clip range                          | 0.2                                            | Maximum policy change per update                                    |
| Epochs / mini-batches               | 5 / 4                                          | How many passes over each batch of experience                       |
| Initial action noise (std)          | 0.2                                            | Exploration noise at the start                                      |
| Entropy bonus                       | 0.005                                          | Keeps some exploration so the noise does not collapse               |
| Value loss coefficient              | 1.0 (clipped)                                  |                                                                     |
| Gradient norm limit                 | 1.0                                            |                                                                     |
| Observation normalisation           | On, for actor and critic                       |                                                                     |
| Checkpoints                         | every 50 iterations                            |                                                                     |

### 3.7 Result

The final training run (`logs/rsl_rl/lift_cube_franka_hx5d20/2026-10-06_12-15-08`, about 10 minutes) learned to lift within 40-60 iterations, and to hold the block at the goal in nearly all episodes from about iteration 300 on. It stayed stable until it was stopped at 700.

Evaluation: **512 deterministic episodes per checkpoint**, using the policy's mean action (no exploration noise) and a fixed seed:

| Checkpoint | Held at goal at end | Lifted at end | Dropped | Distance to goal at end (median) | Tilt of the block at end (median) |
|---|---|---|---|---|---|
| 500 | 99.0 % | 99.2 % | 0 % | 0.1 cm | 39° |
| 550 | 98.6 % | 99.0 % | 0 % | 0.1 cm | 38° |
| 600 | 98.4 % | 98.8 % | 0 % | 0.1 cm | 38° |
| **650 (best)** | **99.4 %** | **99.6 %** | **0 %** | **0.1 cm** | **40°** |
| 700 | 98.6 % | 98.6 % | 0 % | 0.2 cm | 37° |

The differences between checkpoints are within the uncertainty (about ±0.4 %). `model_650` is marginally the best and is the one kept. A video of it is in that run's `videos/play/` folder.

**Behaviour:** the hand reaches the block, closes (the block leaves the table at about 1.2 s), brings it to the goal by about 1.6 s and holds it there for the rest of the episode.

**Weakness:** the hand closes about 5 cm short of the designed grasp point (52 mm), so the block is pinched by the fingertips and held tilted by about 40°. The task metric does not penalise this, since the block's centre still ends up on the goal.

---

## 4. How we got there: problems and fixes

Each row was a real failure, found by measurement (training curves, deterministic evaluation, step-by-step traces of single episodes, or scripted tests) before changing anything.

| #   | Problem                                                        | How it showed up                                                          | Fix                                                                                                                             |
| --- | -------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Robot USD unusable                                             | Joints would not move; "found 2 articulation roots"                       | Default prim set; second articulation root removed                                                                              |
| 2   | Policy flung the block into the air                            | High lift reward, but 89 % of episodes ended with the block off the table | Lift and goal rewards only count while the block is in the hand                                                                 |
| 3   | Exploration too violent                                        | Arm slammed into the block                                                | Smaller arm actions (scale 0.5 → 0.2), smaller starting noise                                                                   |
| 4   | Penalties suddenly took over                                   | Reward or success dropped at about iteration 417 in several runs          | Penalty curriculum removed (it raised penalties 1000× after 10,000 steps)                                                       |
| 5   | Grasp tests were misleading                                    | Grasp held in the air but failed in the task                              | Tests redone with the cube resting on the table                                                                                 |
| 6   | A 6 cm cube cannot be grasped from a flat table with this hand | Scripted grasp: 0 of 5 on the table, 5 of 5 on a 6 cm post                | Object changed to a 6 × 6 × 12 cm block, held on its upper part                                                                 |
| 7   | The arm sagged under gravity                                   | The hand was several cm below its intended pose                           | Gravity turned off for the robot (as in Isaac Lab's high-PD Franka setup)                                                       |
| 8   | A fast hand close pushed the block over                        | Block toppled while the fingers closed                                    | Hand closes over 2 s, all joints together (a per-joint speed limit did not work)                                                |
| 9   | Palm-down grasp fails under exploration noise                  | Scripted grasp with training-level arm noise: 0 of 10                     | Switched to a side grasp (thumb up): 6 of 10 under the same noise                                                               |
| 10  | Policy learned to hover and not touch the block                | Closing reward went to 0; reaching paid safely                            | Rebalanced rewards; toppling no longer ends the episode                                                                         |
| 11  | Block carried to the wrong place                               | Lifted, but held 20-40 cm from the goal                                   | Traced to the wrist (joint 7) sitting at its limit; a different arm solution found with all joints ≥ 0.85 rad from their limits |
| 12  | Training became unstable and collapsed                         | Success rose to 64 %, then fell to 0                                      | Observation normalisation, γ 0.995 → 0.99                                                                                       |
| 13  | Hand closed before reaching the block                          | Trace: closing started at the first step, 6.5 cm away                     | Closing reward only pays near the block                                                                                         |
| 14  | Hand reached the block but never closed                        | Std collapsed to 0.03; hovering paid more                                 | Closing made to pay clearly more than hovering                                                                                  |
| 15  | Hand closed on the block but never lifted                      | Lift reward was all-or-nothing at 2 cm                                    | Lift-progress reward (every mm counts); entropy bonus raised to keep exploring                                                  |

**Lessons:**

- **Test the physics without RL first.** A scripted grasp (move to a pose, close, lift) answers in a minute what a training run answers in twenty, and it separates "impossible" from "not yet learned".
- **Measure what you care about.** `success_rate` looked excellent several times while the block was never held at the goal. A deterministic evaluation with task-specific measures (held at the end, tilt, how far from the block the hand closed) caught each case.
- **Watch for reward loopholes.** The policy repeatedly found the cheapest way to collect reward: flinging the block, hovering safely, holding it on the table, or holding it tilted.
- **Joint limits matter.** The same hand orientation can be reached with very different arm configurations, and one close to a joint limit can make the rest of the task impossible.

---

## 5. Files and tools

| What                                                  | Where                                                                       |
| ----------------------------------------------------- | --------------------------------------------------------------------------- |
| Robot config                                          | `source/isaaclab_assets/isaaclab_assets/robots/franka_hx5d20.py`            |
| Robot USD                                             | `source/isaaclab_assets/data/Robots/Robotis/FrankaHX5D20/franka_hx5d20.usd` |
| Reach task                                            | `source/isaaclab_tasks/isaaclab_tasks/contrib/reach/config/franka_hx5d20/`  |
| Lift task (config, rewards, hand action, PPO)         | `source/isaaclab_tasks/isaaclab_tasks/contrib/lift/config/franka_hx5d20/`   |
| Joint check                                           | `scripts/tools/check_franka_hx5d20_joints.py`                               |
| Hand forward kinematics / posture fit                 | `scripts/tools/fit_franka_hx5d20_grasp.py`                                  |
| Hold test (object pinned, then released)              | `scripts/tools/check_franka_hx5d20_grasp.py`                                |
| Arm pose solver (inverse kinematics in the simulator) | `scripts/tools/find_franka_hx5d20_pregrasp.py`                              |
| Scripted table grasp, single pose                     | `scripts/tools/check_franka_hx5d20_table_grasp.py`                          |
| Scripted table grasp, search over poses and settings  | `scripts/tools/search_franka_hx5d20_table_grasp.py`                         |
| Deterministic evaluation                              | `scripts/tools/evaluate_franka_hx5d20_lift.py`                              |

**Commands** (from the `IsaacLab` folder):

```bash
# train
uv run isaaclab -p scripts/reinforcement_learning/train.py \
  --task Isaac-Lift-Cube-Franka-HX5D20 --num_envs 1024 --max_iterations 1000

# evaluate a checkpoint
uv run isaaclab -p scripts/tools/evaluate_franka_hx5d20_lift.py --num_envs 512 \
  --checkpoint logs/rsl_rl/lift_cube_franka_hx5d20/2026-10-06_12-15-08/model_650.pt

# watch it
uv run isaaclab -p scripts/reinforcement_learning/play.py \
  --task Isaac-Lift-Cube-Franka-HX5D20 --num_envs 4 --checkpoint latest --viz kit

# record a video
uv run --extra video isaaclab -p scripts/reinforcement_learning/play.py \
  --task Isaac-Lift-Cube-Franka-HX5D20 --num_envs 1 --checkpoint latest --video --video_length 400

# training curves
.venv/bin/python -m tensorboard.main --logdir logs/rsl_rl/lift_cube_franka_hx5d20
```

---

## 6. Next steps

The task works. What remains is the quality of the grasp: the block is pinched at the fingertips and tilted, because the hand closes about 5 cm short of the designed grasp point, and because the policy cannot change the shape of the hand.

1. **Better placement with the current hand (smallest step).** Fine-tune from `model_650`, with the closing reward paying only within about 2.5 cm of the grasp point and a small reward for keeping the block upright. In scripted tests, closing at the designed point holds the block upright 10 times out of 10.
2. **A few synergies.** Replace the single open/close value with 2-3 continuous values, each blending in one hand-designed shape: for example finger curl, thumb opposition and finger spread. The policy then decides, within limits, how the thumb and fingers close (7 + 2-3 actions). This is a moderate step up in difficulty, and the natural middle ground.
3. **Every joint.** Let the policy command all 20 hand joints directly (7 + 20 actions). This is the most flexible option, and the only one where the policy can invent its own grasp, but it is a much harder learning problem. It usually needs much more training, careful rewards for finger contacts, or demonstrations to start from.

**Limits of the hand to keep in mind:** the thumb can oppose the fingers by at most about 153° (a human thumb gets close to 180°), with one thumb joint at its limit. In scripted tests, placing the thumb opposite the fingers at the current grasp point made the grasp worse (0-2 of 10 lifts), because the block sits where the opposing thumb wants to go. A true thumb-to-finger pinch would need a different grasp point, found together with the thumb posture.

Other possible extensions: wider block spawn range and random goals, random block size, mass and friction, starting the hand further away, and transferring the policy to the Newton physics backend.
