# tests/test_hx5_d20_patches.py

"""CPU geometry tests only. These DO NOT demonstrate physical Newton contacts."""
import numpy as np
import pytest

from contact_prediction_rl.robots.hx5_d20_patches import (
    FINGERS,
    PatchSelection,
    active_contact_is_inside,
    local_to_world,
    patch_from_bounds,
    world_to_local,
)

# A synthetic box with the *reported index bounds* from the October diagnostic.
# No fake solver constraints or modified observation/force tensors are used.
LO = np.array([-0.0095, -0.0115, -0.00625])
HI = np.array([0.0125, 0.0115, 0.038])


def make(finger="index"):
    return patch_from_bounds(finger, LO, HI, (7,))


def test_five_links_order_normals_and_nonzero_local_boxes():
    patches = [make(f) for f in FINGERS]
    assert [p.link for p in patches] == ["finger_r_link4", "finger_r_link8", "finger_r_link12", "finger_r_link16", "finger_r_link20"]
    assert patches[0].normal_local == (0., 1., 0.)
    assert all(p.normal_local == (1., 0., 0.) for p in patches[1:])
    assert all(np.all(p.dimensions_m > 0) for p in patches)
    assert all(np.all(p.center_local_m <= HI) for p in patches)


def test_index_reported_tip_point_accepted_middle_proximal_example_rejected():
    # Based only on geometric bounding region; NOT a physical validation result.
    assert make().contains_local([0.00898, 0.00200, 0.03686])
    assert not make("middle").contains_local([0.00764, 0.01117, 0.00659])


def test_multiple_active_constraints_or_and_other_pairs_rejected():
    p = make()
    pos = np.array([0.2, 0.3, 0.4])
    q = np.array([0., 0., 0., 1.])
    point = pos + np.array([0.00898, 0.002, 0.03686])
    inside, local = active_contact_is_inside(p, point, pos, q, actual_pair=True, active_constraint=True,
                                            signed_distance_m=-0.0002, max_signed_distance_m=.002)
    assert inside and p.contains_local(local)
    for pair, active, dist in [(False, True, 0), (True, False, 0), (True, True, .005)]:
        inside, _ = active_contact_is_inside(p, point, pos, q, actual_pair=pair,
                                           active_constraint=active, signed_distance_m=dist,
                                           max_signed_distance_m=.002)
        assert not inside


def test_body_rotation_translation_and_tolerance():
    p = make()
    q = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    body = np.array([0.1, -1.0, 2.5])
    local = np.array([0.009, 0., 0.035])
    np.testing.assert_allclose(world_to_local(local_to_world(local, body, q), body, q), local, atol=1e-12)
    assert p.contains_local(p.center_local_m)
    almost = np.asarray(p.upper_local_m) + 0.001
    assert p.contains_local(almost)
    assert not p.contains_local(np.asarray(p.upper_local_m) + 0.003)


def test_configuration_and_geometry_guardrails():
    with pytest.raises(ValueError):
        PatchSelection(inclusion_tolerance_m=-1)
    with pytest.raises(ValueError):
        PatchSelection(ordinary_x_tail=2)
    with pytest.raises(ValueError):
        patch_from_bounds("index", HI, LO, (1,))
    with pytest.raises(ValueError):
        patch_from_bounds("index", LO, HI, ())
    with pytest.raises(ValueError):
        patch_from_bounds("palm", LO, HI, (1,))


def test_compiled_collision_mesh_vertices_have_geom_translation_and_rotation(monkeypatch):
    """Checks the documented geom-frame => distal-body-frame transform.

    Test doubles exercise the mathematics, *not* the Newton integration.
    """
    import sys
    from types import SimpleNamespace

    from contact_prediction_rl.robots.hx5_d20_patches import candidate_patches_from_compiled_mjcf

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_MESH=7)))
    # One identically shaped mesh per body, including thumb. The ordinary
    # fingers must derive identical *bounds*, not merely use copied constants.
    vertices = np.asarray([
        [-.01, -.01, -.006], [-.01, .01, .038],
        [.012, -.01, .038], [.012, .01, -.006],
    ])
    model = SimpleNamespace(
        geom_contype=np.ones(5, dtype=int), geom_conaffinity=np.ones(5, dtype=int),
        geom_type=np.full(5, 7), geom_dataid=np.arange(5),
        mesh_vert=np.vstack([vertices] * 5),
        mesh_vertadr=np.arange(5) * 4, mesh_vertnum=np.full(5, 4),
        geom_pos=np.vstack(([.004, 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.])),
        geom_quat=np.tile([np.sqrt(.5), 0., 0., np.sqrt(.5)], (5, 1)),
    )
    geometry_bodies = np.arange(5)
    tip_bodies = {finger: {i} for i, finger in enumerate(FINGERS)}
    patches = candidate_patches_from_compiled_mjcf(model, geometry_bodies, tip_bodies)
    assert len(patches) == 5
    assert patches["thumb"].collision_bounds_m[0][0] == pytest.approx(-.006)
    assert patches["index"].collision_bounds_m[0][0] == pytest.approx(-.01)
    assert patches["index"].collision_bounds_m[1][1] == pytest.approx(.012)
    assert patches["thumb"].collision_geom_ids == (0,)


def test_collision_nonmesh_is_not_silently_ignored(monkeypatch):
    import sys
    from types import SimpleNamespace

    from contact_prediction_rl.robots.hx5_d20_patches import candidate_patches_from_compiled_mjcf

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_MESH=7)))
    model = SimpleNamespace(geom_contype=np.ones(5), geom_conaffinity=np.ones(5), geom_type=np.zeros(5))
    with pytest.raises(RuntimeError, match="Non-mesh collision geom"):
        candidate_patches_from_compiled_mjcf(model, np.arange(5), {finger: {i} for i, finger in enumerate(FINGERS)})


def test_cpu_mock_newton_event_classification_and_release(monkeypatch):
    """Synthetic solver-buffer test, *not* evidence of a real physics run."""
    import sys
    from types import SimpleNamespace

    import torch

    from contact_prediction_rl.robots.hx5_d20_patches import NewtonPatchDiagnostic
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_MESH=7)))

    class Array:
        def __init__(self, data):
            self.value = np.asarray(data)
        def numpy(self):
            return self.value

    v = np.asarray([
        [-.0095, -.0115, -.00625], [-.0095, .0115, .038],
        [.0125, -.0115, .038], [.0125, .0115, -.00625],
    ])
    model = SimpleNamespace(
        geom_contype=np.ones(6, dtype=int), geom_conaffinity=np.ones(6, dtype=int),
        geom_type=np.full(6, 7), geom_dataid=np.arange(6),
        mesh_vert=np.vstack([v]*6), mesh_vertadr=np.arange(6)*4,
        mesh_vertnum=np.full(6, 4), mesh_faceadr=np.zeros(6, dtype=int),
        mesh_facenum=np.ones(6, dtype=int), mesh_face=np.array([[0, 1, 2]]),
        geom_pos=np.zeros((6, 3)), geom_quat=np.tile([1., 0., 0., 0.], (6, 1)),
    )
    positions = np.array([[.00898, .00200, .03686], [.00764, .01117, .00659], [0., 0., 0.]])
    contact = SimpleNamespace(
        geom=Array([[1, 5], [2, 5], [1, 5]]), efc_address=Array([[0], [1], [2]]),
        pos=Array(positions), dist=Array([-.0002, -.0003, -.0001]),
    )
    mjw = SimpleNamespace(nacon=Array([3]), contact=contact)
    solver = SimpleNamespace(mj_model=model, mjw_data=mjw)
    probe = SimpleNamespace(
        solver=solver, geometry_bodies=np.arange(6), object_bodies={5},
        tip_bodies={finger: {i} for i, finger in enumerate(FINGERS)},
    )
    poses = torch.zeros(1, 5, 7)
    poses[..., 6] = 1.
    robot = SimpleNamespace(body_names=["finger_r_link4", "finger_r_link8", "finger_r_link12", "finger_r_link16", "finger_r_link20"],
                            data=SimpleNamespace(body_link_pose_w=poses))
    env = SimpleNamespace(scene={"robot": robot})
    diagnostic = NewtonPatchDiagnostic(probe, env)
    bits, events = diagnostic.sample(env)
    assert bits.tolist() == [0, 1, 0, 0, 0]
    assert len(events) == 3
    assert [ev["inside"] for ev in events] == [True, False, False]
    assert events[1]["finger"] == "middle"
    mjw.nacon = Array([0])
    bits, events = diagnostic.sample(env)
    assert bits.tolist() == [0, 0, 0, 0, 0] and not events


@pytest.mark.parametrize('angle', [0., np.pi / 2, np.pi, .71])
@pytest.mark.parametrize('axis', np.eye(3))
def test_xyzw_known_rotations_and_original_diagnostic_agree(angle, axis):
    import torch

    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    q = np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)]
    body = np.array([.2, -.4, 1.1])
    local = np.array([.008979, .002, .036858])
    tq = torch.tensor(q).reshape(1, 4)
    world = body + quat_apply(tq, torch.tensor(local).reshape(1, 3))[0].numpy()
    np.testing.assert_allclose(local_to_world(local, body, q), world, atol=1e-12)
    np.testing.assert_allclose(world_to_local(world, body, q), local, atol=1e-12)
    original = quat_apply_inverse(tq, torch.tensor(world-body).reshape(1, 3))[0].numpy()
    np.testing.assert_allclose(world_to_local(world, body, q), original, atol=1e-12)


def test_mujoco_wxyz_rotation_matches_installed_api():
    import mujoco

    from contact_prediction_rl.robots.hx5_d20_patches import wxyz_to_rotation
    q = np.array([.5, .5, .5, .5])
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, q)
    np.testing.assert_allclose(wxyz_to_rotation(q), matrix.reshape(3, 3), atol=1e-12)


@pytest.mark.parametrize('finger,point', [
    ('thumb', [.009774, .048532, .004436]),
    ('index', [.008979, .002000, .036858]),
    ('middle', [.008979, .002000, .036855]),
    ('ring', [.008994, .002000, .036869]),
    ('little', [.009004, .002000, .036883]),
])
def test_previous_physical_first_contact_references(finger, point):
    # Recorded local coordinates test geometry only; no simulated events here.
    p = (patch_from_bounds(finger, np.array([-.011, -.009482268, -.0095]),
                           np.array([.011, .0535, .0125]), (15,))
         if finger == 'thumb' else make(finger))
    assert p.contains_local(point)
    q = np.array([.2, -.3, .4, .5])
    q /= np.linalg.norm(q)
    body = np.array([.1, .3, -.2])
    np.testing.assert_allclose(world_to_local(local_to_world(point, body, q), body, q), point, atol=1e-12)


def test_visual_markers_follow_live_xyzw_pose_and_clear():
    from types import SimpleNamespace

    from contact_prediction_rl.robots.hx5_d20_patches import NewtonPatchDiagnostic

    class Marker:
        def visualize(self, **values):
            self.values = {k: v.cpu().numpy() for k, v in values.items()}
        def set_visibility(self, value):
            self.visible = value

    diagnostic = NewtonPatchDiagnostic.__new__(NewtonPatchDiagnostic)
    diagnostic.patches = {'index': make()}
    diagnostic.link_indices = {'index': 0}
    diagnostic.collision_vertex_samples = {'index': np.array([[1., 0., 0.]])}
    green, inside, outside, vertices = [Marker() for _ in range(4)]
    diagnostic._markers = ({'index': green}, {'inside': inside, 'outside': outside}, vertices)
    poses = np.array([[.2, .3, .4, 0., 0., np.sqrt(.5), np.sqrt(.5)]])
    events = [{'world_m': [1., 2., 3.], 'inside': True},
              {'world_m': [4., 5., 6.], 'inside': False}]
    env = SimpleNamespace(device='cpu')
    diagnostic._render(env, poses, events)
    np.testing.assert_allclose(vertices.values['translations'], [[.2, 1.3, .4]], atol=1e-7)
    expected = local_to_world(make().center_local_m, poses[0, :3], poses[0, 3:])
    np.testing.assert_allclose(green.values['translations'][0], expected, atol=1e-7)
    np.testing.assert_allclose(green.values['orientations'][0], poses[0, 3:], atol=1e-7)
    assert inside.visible and outside.visible
    poses[0, :3] += 1
    diagnostic._render(env, poses, [])
    np.testing.assert_allclose(green.values['translations'][0], expected + 1, atol=1e-7)
    assert not inside.visible and not outside.visible
