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
4. **Scripted rotation:** Reinitialises the selected grasp and tests several
   small finger-only joint-position trajectories. Reports measured signed
   rotation around the stationary palm's local +X axis (world vertical),
   translation drift, contact persistence, and orientation error against 15°
   and 45° goals. No cube pose is written during the hold/rotation phases.

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
