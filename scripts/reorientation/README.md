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

1. **Grasp acquisition:** Uses the already validated Newton fingers-down
   Franka pose, tries candidate positions of a 5 cm *dynamic* cube in the
   panda_hand frame, and closes the 20 finger joints using Lift or the existing
   single-finger diagnostic postures. A narrow kinematic pedestal supports
   the cube **only during closure**.
2. **Unsupported hold:** Removes the pedestal without moving the cube.
   Keeps the arm fixed and checks for two seconds that cube translation is
   limited and both thumb and another finger produce simultaneous positive
   geometric patch-contact labels.
3. **Scripted rotation:** Reinitialises the selected grasp and tests several
   small finger-only joint-position trajectories. Reports measured signed
   rotation around the stationary palm's local +X axis in world coordinates,
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
