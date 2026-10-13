# Lift task: letting the policy shape the hand (hand synergies)

*Status: synergy results and the control experiment (section 6) are finished. Next: control of every finger joint.*

## Summary

In report 1 the hand was a gripper with one value: open or closed. All 20 finger joints followed one fixed path, so the policy never chose the shape of the hand. Here we gave the policy three values instead of one, each blending in a hand-designed finger shape (a "synergy"): **curl**, **thumb** and **spread**.

What we found, in short:

- **One run is the best grasp we have seen.** With the synergy task, seed 4 holds the block at the goal in **100 %** of 512 test episodes, and holds it **upright** (median tilt **7°**, 96 % under 10°). The best run with the one-value hand holds it at a tilt of 21° and none of its episodes are upright.
- **But it barely uses the new freedom.** At the end of the episode, curl is fully closed in every episode, **thumb is open (about 0) and spread is 0**. The policy mostly found "close the fingers, keep the thumb open".
- **The synergy task is not more reliable.** After 300 iterations, 1 of 5 seeds ends up holding the block at the goal (against 3 of 5 for the one-value hand). Two synergy runs found a good grasp and then lost it.
- **The control says most of this is the thumb-open posture, not learned control.** A hand with a single open/close value whose closed posture is "fingers curled, thumb open" gets a similar grasp in the seeds that work (100 % held, tilt 13°, closes 46 mm from the grasp point). The synergy hand's best run is somewhat better (tilt 7°, 43 mm), but the control's runs vary from 10° to 25° tilt, so we cannot call that difference real.
- **So we cannot claim the three synergies help yet.** What the experiments did show: keeping the thumb open gives a good grasp, and spread is never used.

## 1. What changed

### The hand action

| | One-value hand (report 1) | Synergy hand (this report) |
|---|---|---|
| Hand actions | 1 | 3 |
| Total actions | 7 arm + 1 | 7 arm + 3 |
| What the policy chooses | open or close | how much to close each of three shapes |

The three shapes, each a fixed set of finger joint angles:

| Synergy | What moves | Meaning |
|---|---|---|
| **curl** | bending joints of index, middle, ring and little finger | close the four fingers |
| **thumb** | the four thumb joints | bring the thumb across and bend it |
| **spread** | the sideways joint of each finger | fan the fingers apart (outer fingers move twice as much as inner ones) |

Each synergy has a **closure** between 0 (open) and 1 (fully closed). The hand target is the open hand plus closure × shape, summed over the three. A negative action closes a synergy and a non-negative one opens it, as in the old binary action. Closures move at a limited speed (a full close takes 2 s), because a fast close knocks the block over (report 1, problem 8).

Curl and thumb at closure 1 together are exactly the old "closed" posture. So the new hand can do everything the old one did, and more.

### Everything else is unchanged
Same robot, block, goal, rewards, episode length and PPO settings. The "closed hand near the block" reward now counts the **curl** closure. The only other difference is that the network sees 2 more numbers (the last action is now 10 values instead of 8).

## 2. Check before training: scripted grasps

Before training anything, we tried grasps with a script (move to the block, close, lift), as in report 1. We tried 27 combinations of closures (curl 0.6 / 0.8 / 1.0, thumb 0 / 0.5 / 1, spread 0 / 0.5 / 1), 5 block positions each.

| Region | Result |
|---|---|
| curl 0.6 (any thumb or spread) | 0 of 5 lifted: fingers never close enough |
| curl 0.8 | works with thumb 0–0.5; fails with full thumb and spread |
| curl 1.0 (any thumb, any spread) | 5 of 5 lifted, block upright (tilt 1–5°) |

With a little arm noise (0.03 rad, like training exploration), 10 trials each:

| curl, thumb, spread | lifted | upright |
|---|---|---|
| 1, 0.5, 0.5 | 10 / 10 | 10 / 10 |
| 1, 0, 0 | 9 / 10 | 8 / 10 |
| 1, 1, 0 (the old closed hand) | 6 / 10 | 6 / 10 |
| 1, 1, 1 | 1 / 10 | 1 / 10 |

So a good grasp exists in the synergy space, and "thumb open" works at least as well as "thumb closed". Scripted grasps are upright (1–3°), while the trained policy of report 1 held the block at about 40°. The difference is in how the policy places the hand, not in what the hand can do.

## 3. Training results (5 seeds each, 300 iterations, 1024 environments)

A seed is the random starting point of a run. This task is very sensitive to it (report 2), so single runs say little. Numbers are the mean over the last 20 iterations; "held" is the share of episodes that end with the block held within 5 cm of the goal.

| Seed | One-value hand: reward / held | Synergy hand: reward / held |
|---|---|---|
| 1 | 173 / 0.77 | 62 / 0.00 |
| 2 | 7 / 0.00 | 5 / 0.00 |
| 3 | 226 / 0.95 | 7 / 0.00 |
| 4 | 14 / 0.00 | **234 / 1.00** |
| 42 | 207 / 0.85 | 47 / 0.00 |
| **Runs ending up holding the goal** | **3 of 5** | **1 of 5** |

The runs fall into three groups:

- **Never closes the hand** (one-value seeds 2 and 4, synergy seeds 2 and 3). The policy hovers next to the block and never grasps, the same "hover and don't touch" failure as problem 10 in report 1.
- **Finds the grasp, then loses it** (synergy seeds 1 and 42). Seed 1 reached 0.91 around iteration 150 and ended at 0 held. The one-value hand had one mild case of this (seed 1, 1.00 down to 0.77).
- **Works** (one-value seeds 1, 3, 42, synergy seed 4).

With 5 seeds each, "3 of 5 against 1 of 5" is **not a statistically meaningful difference**. It is a hint.

## 4. The best synergy policy, evaluated

Evaluation: 512 deterministic episodes (the policy's mean action, no exploration noise), fixed seed.

| Policy | Held at goal | Tilt (median) | Upright (<10°) | Distance to goal | Hand closes this far from the grasp point |
|---|---|---|---|---|---|
| **Synergy seed 4** | **100 %** | **7°** | **96 %** | 0.3 cm | **43 mm** |
| One-value hand, seed 3 (best) | 100 % | 21° | 0 % | 0.3 cm | 51 mm |
| One-value hand, report 1 | 99.4 % | 40° | – | 0.1 cm | 52 mm |
| Synergy seed 1, final | 0 % | 10° | 52 % | 9 cm | 68 mm |
| Synergy seed 42, final | 0 % | 27° | 0 % | 17 cm | 72 mm |

What the synergy seed 4 policy does at the end of the episode: curl **1.00**, thumb **0.04** (closed in only a few episodes), spread **0.00**. A video of it is in the run folder (see section 7).

What the numbers say:

- Its grasp is better in the way report 1 wanted: **more upright and closer to the intended grasp point**.
- The failing synergy runs use the **same finger shape** (curl on, thumb and spread off). They fail because the hand closes about 7 cm from the grasp point, so the block is pinched and dragged. That is the weakness described in report 1, not a problem of the new action.
- **Spread was never used** in any evaluated policy.

One thing we do not understand yet: seed 1 at iteration 150 held the block at the goal in 91 % of *training* episodes, but only 12 % in the deterministic test of the same checkpoint. For the one-value hand the two agree. This policy may rely on its exploration noise.

## 5. What we can and cannot say (before the control)

**We can say**
- A good grasp exists in the synergy space (scripted tests), and one trained policy found it.
- That policy is better than the one-value policies in tilt and in closing point, with one run each.
- The extra freedom was used in one way only: keeping the thumb open.

**We could not say yet** (section 6 answers the first of these)
- That the synergies are what made the grasp better, rather than a better fixed posture.
- That synergies are as reliable to train as the one-value hand (the numbers suggest not, but 5 seeds is too few).
- Why two runs lose the grasp after finding it.

## 6. The control experiment

**Question:** is the better grasp a result of *learned control*, or just of a *better fixed posture*?

**Control:** the one-value hand (open or closed), but where "closed" means **fingers curled, thumb open**. The policy has no choice of finger shape at all. Same task, rewards and settings, 5 seeds, 300 iterations.

### Results (control, thumb open, one value)

| Seed | Reward / held (training, last 20 its.) | Held at goal (test) | Lifted | Tilt | Upright | Closes from grasp point |
|---|---|---|---|---|---|---|
| 1 | 5 / 0.00 | 0 % | 0 % | – | – | 83 mm (never grasps) |
| 2 | 233 / 1.00 | **100 %** | 100 % | 13° | 2 % | 46 mm |
| 3 | 25 / 0.00 | 0 % | 0 % | – | – | 83 mm (never grasps) |
| 4 | 74 / 0.06 | 0.2 % | 76 % | 10° | 44 % | 56 mm (90th pct 79 mm) |
| 42 | 148 / 0.61 | **93 %** | 100 % | 25° | 0 % | 47 mm |

Seeds 1 and 3 never close the hand (the hover failure again). Seed 4 lifts the block but carries it about 10 cm from the goal, as synergy seeds 1 and 42 did.

### The three setups side by side

| Setup | Holds the goal at the end | Never closes the hand | Best grasp (tilt / upright / closing point) |
|---|---|---|---|
| One value, thumb closed (report 1 posture) | 3 of 5 | 2 of 5 | 21° / 0 % / 51 mm |
| **One value, thumb open (control)** | 2 of 5 | 2 of 5 | 13° / 2 % / 46 mm |
| Three synergies | 1 of 5 | 2 of 5 | 7° / 96 % / 43 mm |

### What it means

| Prediction made before the experiment | What happened |
|---|---|
| If the control gives the same grasp, the synergies add nothing yet | **Mostly this.** The control reaches 100 % held at 13° tilt and 46 mm, near the synergy's 7° and 43 mm. |
| If the control is clearly worse, the learned choice matters | **Not clearly.** The synergy run is better (7° against 13°, 25° and 10°), but that is one synergy run against three control runs that vary by 15°. |

- **Main finding:** the better grasp comes mainly from **keeping the thumb open**. The extra freedom of the synergy hand, as designed, did not show a clear benefit.
- **Reliability is not a result of the synergies.** About 2 of 5 seeds never close the hand in *every* setup, and 1–3 of 5 finish holding the block. With 5 seeds each, the differences between setups are within chance.
- **Tilt differs a lot between seeds** within the same setup (control: 10°–25°; synergy: 7°–27°). A single run is not a measure of a setup.
- **A remaining open question:** why does the best synergy run reach 96 % upright when the control's best reaches 44 %? It could be luck, or something the synergy policy learned about hand placement. More seeds are needed.

### Honest limits
- 5 seeds per setup, 300 iterations, one evaluation per run.
- "Held" at the end of 300 iterations is a snapshot. Some runs were still changing (see the lost grasps in section 3).

## 7. Files and commands

| What | Where |
|---|---|
| Hand shapes | `src/contact_prediction_rl/robots/franka_hx5d20.py` (`FRANKA_HX5D20_HAND_SYNERGIES`) |
| Synergy action | `src/contact_prediction_rl/tasks/lift/mdp/actions.py` (`SynergyJointPositionAction`) |
| Task ID | `ContactPredictionRl-Lift-Block-Franka-HX5-D20-Synergy-Right` |
| Control task ID (thumb open, one value) | `ContactPredictionRl-Lift-Block-Franka-HX5-D20-Curl-Right` |
| Control evaluations (full output) | `logs_exp/lorenzo/task_lift_cube/curl_evals.txt` |
| Scripted grasp search with synergies | `scripts/franka_hx5d20/search_franka_hx5d20_table_grasp.py --synergies "curl,thumb,spread;..."` |
| Best run | `logs/ppo/lift_block_franka_hx5d20_synergy/2026-10-12_15-37-41_syn_seed4/` |
| Video of the best policy | `logs/ppo/lift_block_franka_hx5d20_synergy/2026-10-12_15-37-41_syn_seed4/videos/play/clip_model_299_0000.mp4` |

```bash
# train (5 seeds in report: 1, 2, 3, 4, 42)
uv run --extra isaacsim python scripts/train.py \
  --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Synergy-Right \
  --num_envs 1024 --max_iterations 300 --seed 4

# evaluate (also prints the final closure of each synergy)
uv run --extra isaacsim python scripts/franka_hx5d20/evaluate_franka_hx5d20_lift.py \
  --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Synergy-Right --num_envs 512 \
  --checkpoint logs/ppo/lift_block_franka_hx5d20_synergy/2026-10-12_15-37-41_syn_seed4/model_299.pt

# record a video
uv run --extra isaacsim --extra video python scripts/play.py \
  --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Synergy-Right \
  --num_envs 1 --video --video_length 400 \
  --checkpoint logs/ppo/lift_block_franka_hx5d20_synergy/2026-10-12_15-37-41_syn_seed4/model_299.pt
```

Note: all of this uses the new local PPO implementation (not RSL-RL), so results are not directly comparable to checkpoints trained in report 1. The one-value hand was retrained with the new PPO for the comparison above.
