# Newton free-cube reorientation feasibility

This is a standalone **diagnostic**, not a training task or a demonstrated
working dexterous controller. It does not modify the existing Reorientation
observation groups (40D policy, 5D privileged contact), 20 finger actions,
reward, termination, contact patches, PPO or auxiliary head.

## Run

From the repository root:

~~~bash
git switch feature/reorientation-feasibility
git pull --ff-only origin feature/reorientation-feasibility
uv run pytest -m "not gpu" tests/test_reorientation_feasibility.py -v
uv run python scripts/reorientation/validate_scripted_rotation.py --visualizer newton_gl
~~~

Headless, with fewer search attempts:

~~~bash
uv run python scripts/reorientation/validate_scripted_rotation.py \
  --max-grasps 6 --max-rotation-trials 8
~~~

## What the experiment does

1. **Palm-up arm pose:** Uses the *compiled Newton/MuJoCo kinematics* to solve
   the seven Franka joint angles once. The HX5 palm normal (local +X) must
   point world +Z, with the finger axis (local +Z) approximately world +X.
   The solution is written into reset joint positions and held by the Franka
   joint-position actuators; the arm is never an action output.
2. **Grasp acquisition:** Places a free dynamic 5 cm cube over the upward
   facing palm, then tries the existing Lift and diagnostic hand postures.
   The default leaves the pedestal parked out of the scene so the cube rests
   directly in the palm. `--use-pedestal` temporarily supports it during
   closure if the unsupported setup proves infeasible.
3. **Unsupported hold:** Checks two seconds of cube retention, arm-pose
   stability, and simultaneous thumb plus other fingertip patch contacts.
   When enabled, the temporary support is retracted before this check.
4. **Scripted rotation:** Reinitialises the selected grasp and tests a staged
   finger gait: unload an index/middle finger while retaining thumb opposition,
   apply a tangential joint-position sweep, regrasp, then recover. Continuous
   interpolation connects each phase. The old static-offset trajectory can
   still be selected for comparison. Measured cube rotation is reported about
   the stationary palm's local +X axis (world vertical), with retention,
   contact persistence and errors against 15° / 45° goals. No cube pose is
   written during hold or rotation.

Success requires a free, retained cube and orientation error within 5°.
These goal quaternions are **diagnostic only**, not policy observations.
The evaluator reports physical quaternion orientation, without treating a
cube's visually equivalent 90° rotations as interchangeable.

Output markers: GRASP_TRIAL, GRASP_SELECTED, ROTATION_TRIAL and
FEASIBILITY_SUMMARY. FEASIBILITY_INCOMPLETE means no successful physical
demonstration was established; share the full output so the setup can be
improved. This script does *not* guarantee an existing feasible grasp.

## Optional support and diagnostics

```bash
# Use only if gravity causes the free cube to roll/fall out during acquisition:
uv run python scripts/reorientation/validate_scripted_rotation.py \
  --use-pedestal --visualizer newton_gl
```

The script emits `PALM_UP_FK_CHECK`, `PALM_UP_IK_CANDIDATE`,
`PALM_UP_IK_SELECTED` and `PALM_UP_ALIGNMENT`. It **refuses** an arm IK
solution whose orientation, position or collision checks fail; no fallback
to fingers-down is permitted. The palm orientation is rechecked after every
reset and after closure. Robot-arm drift is checked independently of object
rotation. The 20D finger action and 40D/5D observation contracts remain
unchanged. This is still a physics experiment, not a finished RL task.

## Revised grasp placement and scripted finger gait

The candidate list now searches more central palm positions first (less
longitudinal +Z extension, lateral Y close to zero). It retains the previously
successful `diagnostic@0.050,0.025,0.100` candidate among the first six
posture trials as a fallback. For each grasp the log includes the **measured**
`cube_palm_local_m` after physics settling, so we can distinguish initial
placement from the actual stable contact configuration. Central placement is
a hypothesis and may not succeed.

The palm-up inverse-kinematics solve constrains all seven Franka joints to
remain at least **0.15 rad** from their joint limits by default. It prints
`joint_limit_margin_rad` and refuses to use an infeasible pose. If necessary,
change this independently of the hand controller:

```bash
uv run python scripts/reorientation/validate_scripted_rotation.py \
  --arm-limit-margin-rad 0.10 --visualizer newton_gl
```

Run a complete staged finger gait with live Newton physics:

```bash
uv run python scripts/reorientation/validate_scripted_rotation.py \
  --controller gait --gait-cycles 1 --visualizer newton_gl
```

Use `--controller offset` to repeat the earlier one-way offset baseline, or
`--controller both --max-rotation-trials 36` to evaluate both sets.
The `GAIT_PHASE` lines show actual rotation after unload, sweep, regrasp and
recovery. A simulated joint motion is **not** counted as object rotation;
success still requires retaining the free cube and reaching the target
orientation.

**This change does not establish a successful rotation in advance.** Only
the user's GPU Newton run can validate the safer arm pose, new cube placement
and whether the finger gait produces more rotation.
