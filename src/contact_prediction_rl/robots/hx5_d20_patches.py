# src/contact_prediction_rl/robots/hx5_d20_patches.py

"""Candidate HX5-D20 *geometric* contact regions, not calibrated tactile sensors.

No Isaac Lab / Newton import: geometry classification can be tested on CPU.
The mesh bounds come from the compiled MuJoCo collision meshes actually used
by Newton MJWarp, not from visual geometry or hardcoded CAD coordinates.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from contact_prediction_rl.robots.config import hx5_d20_patches as PATCH_CFG

FINGERS = ("thumb", "index", "middle", "ring", "little")
DISTAL_LINKS = dict(zip(FINGERS, ("finger_r_link4", "finger_r_link8", "finger_r_link12", "finger_r_link16", "finger_r_link20"), strict=True))


@dataclass(frozen=True)
class PatchSelection:
    """Fractions of the observed collision AABB, not anatomical measurements.

    Ordinary fingers: +X-facing distal region, keeping the distal 38% of +Z,
    the high 38% of X and middle 84% of Y.
    Thumb: +Y distal cap region, keeping the distal 35% of +Y and middle
    80% of each transverse X and Z dimension. The thumb normal is a distal-
    cap proxy and MUST NOT be claimed to coincide with its actual tactile pad.
    """
    ordinary_x_tail: float = PATCH_CFG.ORDINARY_X_TAIL
    ordinary_z_tail: float = PATCH_CFG.ORDINARY_Z_TAIL
    ordinary_y_central: float = PATCH_CFG.ORDINARY_Y_CENTRAL
    thumb_y_tail: float = PATCH_CFG.THUMB_Y_TAIL
    thumb_x_central: float = PATCH_CFG.THUMB_X_CENTRAL
    thumb_z_central: float = PATCH_CFG.THUMB_Z_CENTRAL
    inclusion_tolerance_m: float = PATCH_CFG.INCLUSION_TOLERANCE_M
    max_signed_distance_m: float = PATCH_CFG.MAX_SIGNED_DISTANCE_M

    def __post_init__(self) -> None:
        for name in ("ordinary_x_tail", "ordinary_z_tail", "ordinary_y_central", "thumb_y_tail", "thumb_x_central", "thumb_z_central"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.inclusion_tolerance_m < 0 or self.max_signed_distance_m < 0:
            raise ValueError("Tolerances must be nonnegative")


_DEFAULT_SELECTION = PatchSelection()


@dataclass(frozen=True)
class CandidatePatch:
    """Axis-aligned candidate box in a distal rigid-body frame, with metre units."""

    finger: str
    link: str
    lower_local_m: tuple[float, float, float]
    upper_local_m: tuple[float, float, float]
    normal_local: tuple[float, float, float]
    inclusion_tolerance_m: float
    collision_geom_ids: tuple[int, ...]
    collision_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    rationale: str

    @property
    def center_local_m(self) -> np.ndarray:
        return (np.asarray(self.lower_local_m) + self.upper_local_m) / 2

    @property
    def dimensions_m(self) -> np.ndarray:
        return np.asarray(self.upper_local_m) - self.lower_local_m

    @property
    def orientation_local_wxyz(self) -> tuple[float, float, float, float]:
        return (1., 0., 0., 0.)  # axis-aligned proxy volume

    def contains_local(self, point_local_m: np.ndarray) -> bool:
        point = np.asarray(point_local_m, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            return False
        tol = self.inclusion_tolerance_m
        return bool(np.all(point >= np.asarray(self.lower_local_m) - tol)
                    and np.all(point <= np.asarray(self.upper_local_m) + tol))


def _central(lo: float, hi: float, fraction: float) -> tuple[float, float]:
    margin = (hi - lo) * (1 - fraction) / 2
    return lo + margin, hi - margin


def patch_from_bounds(finger: str, xyz_min: np.ndarray, xyz_max: np.ndarray,
                      collision_geom_ids: tuple[int, ...], cfg: PatchSelection = _DEFAULT_SELECTION) -> CandidatePatch:
    """Derive a candidate box from configured fractions of body-frame mesh bounds.

    This is a REPRODUCIBLE CANDIDATE only. A box intersects the underlying
    active collision surface; it is *not* itself a collider or a sensing pad.
    """
    if finger not in FINGERS:
        raise ValueError(f"Unknown fingertip: {finger}")
    lo = np.asarray(xyz_min, dtype=float)
    hi = np.asarray(xyz_max, dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or not (np.isfinite(lo).all() and np.isfinite(hi).all()) or np.any(hi <= lo):
        raise ValueError("Expected finite, non-degenerate 3D collision-mesh bounds")
    if not collision_geom_ids:
        raise ValueError("No real collision mesh geoms given")
    lower, upper = lo.copy(), hi.copy()
    if finger == "thumb":
        lower[1] = hi[1] - cfg.thumb_y_tail * (hi[1] - lo[1])
        lower[0], upper[0] = _central(lo[0], hi[0], cfg.thumb_x_central)
        lower[2], upper[2] = _central(lo[2], hi[2], cfg.thumb_z_central)
        normal = (0., 1., 0.)
        rationale = "+Y distal-cap candidate from thumb collision mesh; *not* mapped to thumb tactile array"
    else:
        lower[0] = hi[0] - cfg.ordinary_x_tail * (hi[0] - lo[0])
        lower[2] = hi[2] - cfg.ordinary_z_tail * (hi[2] - lo[2])
        lower[1], upper[1] = _central(lo[1], hi[1], cfg.ordinary_y_central)
        normal = (1., 0., 0.)
        rationale = "+X-side distal +Z candidate (hand curls towards +X); not a calibrated tactile pad"
    return CandidatePatch(
        finger=finger, link=DISTAL_LINKS[finger],
        lower_local_m=tuple(float(x) for x in lower),
        upper_local_m=tuple(float(x) for x in upper),
        normal_local=normal, inclusion_tolerance_m=cfg.inclusion_tolerance_m,
        collision_geom_ids=collision_geom_ids,
        collision_bounds_m=(tuple(float(x) for x in lo), tuple(float(x) for x in hi)),
        rationale=rationale,
    )



def patch_surface_samples(vertices_body: np.ndarray, faces: np.ndarray, patch: CandidatePatch, *, max_points: int = 12) -> np.ndarray:
    """Mesh-surface points inside a fingertip patch, not arbitrary distal vertices.

    Interpolate triangle interiors: a narrow patch may contain no mesh vertices.
    Coordinates must be expressed in the distal body's local frame.
    """
    vertices = np.asarray(vertices_body, dtype=float)
    triangles = np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Expected mesh vertices [N, 3] and faces [M, 3]")
    if max_points < 1 or not len(vertices) or not len(triangles) or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("Invalid triangle mesh or max_points")
    tri = vertices[triangles]
    weights = np.array([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [.5, .5, 0.], [.5, 0., .5], [0., .5, .5],
        [1/3, 1/3, 1/3], [.5, .25, .25], [.25, .5, .25], [.25, .25, .5],
    ])
    samples = np.einsum("tvc,sv->tsc", tri, weights).reshape(-1, 3)
    inside = np.array([p for p in samples if patch.contains_local(p)])
    if not len(inside):
        return np.empty((0, 3))
    ordered = inside[np.argsort(np.linalg.norm(inside - patch.center_local_m, axis=1), kind="stable")]
    selected = []
    for point in ordered:
        if all(np.linalg.norm(point - old) > .0015 for old in selected):
            selected.append(point)
        if len(selected) == max_points:
            break
    return np.asarray(selected)


def wxyz_to_rotation(q: np.ndarray) -> np.ndarray:
    """Column-vector rotation matrix for a unit [w,x,y,z] quaternion."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Expected finite quaternion [w,x,y,z]")
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise ValueError("Zero-length quaternion")
    w, x, y, z = q / norm
    return _rotation_components(x, y, z, w)


def _rotation_components(x, y, z, w) -> np.ndarray:
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def xyzw_to_rotation(q: np.ndarray) -> np.ndarray:
    """Column-vector rotation for Isaac Lab 3 live body poses (XYZW)."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Expected quaternion [x,y,z,w]")
    if not np.isfinite(q).all():
        raise ValueError("Expected finite quaternion [x,y,z,w]")
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise ValueError("Zero-length quaternion")
    return _rotation_components(*(q / norm))


def world_to_local(position_w: np.ndarray, body_position_w: np.ndarray, body_quat_xyzw: np.ndarray) -> np.ndarray:
    """Transform a world point into the live Isaac Lab link frame."""
    return xyzw_to_rotation(body_quat_xyzw).T @ (np.asarray(position_w) - np.asarray(body_position_w))


def local_to_world(position_local: np.ndarray, body_position_w: np.ndarray, body_quat_xyzw: np.ndarray) -> np.ndarray:
    """Transform a link-frame point to world coordinates using an XYZW body pose."""
    return np.asarray(body_position_w) + xyzw_to_rotation(body_quat_xyzw) @ np.asarray(position_local)


def active_contact_is_inside(patch: CandidatePatch, point_world: np.ndarray,
                             body_pos_world: np.ndarray, body_quat_world: np.ndarray,
                             *, actual_pair: bool, active_constraint: bool,
                             signed_distance_m: float, max_signed_distance_m: float) -> tuple[bool, np.ndarray]:
    """Geometric AND active-constraint label; no contact-force interpretation.

    An active constraint may have a slightly positive distance because of the
    solver contact margin. Count only current events within this tolerance;
    *never* latch previous-frame contacts. For several points, caller ORs bits.
    """
    # Live Isaac Lab body poses use XYZW while compiled MuJoCo geom poses use WXYZ
    local = world_to_local(point_world, body_pos_world, body_quat_world)
    accepted = (actual_pair and active_constraint and np.isfinite(signed_distance_m)
                and signed_distance_m <= max_signed_distance_m
                and patch.contains_local(local))
    return bool(accepted), local


def candidate_patches_from_compiled_mjcf(model, geometry_bodies: np.ndarray,
                                        tip_bodies: Mapping[str, set[int]],
                                        cfg: PatchSelection = _DEFAULT_SELECTION) -> dict[str, CandidatePatch]:
    """Read compiled COLLISION mesh vertices in each distal rigid body's frame.

    MuJoCo's compiled mesh vertices are in the geom frame. model.geom_{pos,quat}
    maps that geom frame into its owning rigid body frame. Reject other geom
    types: a geometry-only inference must not silently ignore active colliders.
    """
    # Defer MuJoCo loading so pure geometry helpers remain usable without a simulator
    import mujoco

    result = {}
    for finger in FINGERS:
        ids = [g for g, body in enumerate(geometry_bodies)
               if int(body) in tip_bodies[finger]
               and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
        if not ids:
            raise RuntimeError(f"No collision-enabled distal geom for {finger}")
        vertex_sets = []
        for g in ids:
            if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise RuntimeError(f"Non-mesh collision geom {g} on {finger}; inspect manually")
            m = int(model.geom_dataid[g])
            begin, count = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
            if count < 4:
                raise RuntimeError(f"Insufficient mesh vertices: {finger} geom {g}")
            vertices_geom = np.asarray(model.mesh_vert[begin:begin + count], dtype=float)
            vertices_body = (vertices_geom @ wxyz_to_rotation(model.geom_quat[g]).T
                             + np.asarray(model.geom_pos[g], dtype=float))
            vertex_sets.append(vertices_body)
        vertices = np.concatenate(vertex_sets)
        result[finger] = patch_from_bounds(finger, vertices.min(axis=0), vertices.max(axis=0), tuple(ids), cfg)
    return result


def equivalent_ordinary_collision_bounds(patches: Mapping[str, CandidatePatch], atol_m: float = 1e-5) -> bool:
    """Report equality of derived bounds; never assume geometry equality."""
    ref = np.asarray(patches["index"].collision_bounds_m)
    return all(np.allclose(np.asarray(patches[f].collision_bounds_m), ref, atol=atol_m, rtol=0)
               for f in ("middle", "ring", "little"))



def sampled_collision_mesh_vertices(model, geometry_bodies: np.ndarray,
                                    tip_bodies: Mapping[str, set[int]],
                                    max_vertices_per_geom: int = 80) -> dict[str, np.ndarray]:
    """Read vertex samples of *compiled Newton collision meshes*, not visuals.

    Markers show sampled collision vertices, not a reconstructed triangulated
    collision surface and not the robot's USD render mesh.
    """
    out = {}
    for finger in FINGERS:
        samples = []
        for g, body in enumerate(geometry_bodies):
            if int(body) not in tip_bodies[finger] or not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
                continue
            m = int(model.geom_dataid[g])
            start, count = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
            pick = np.linspace(0, count-1, min(count, max_vertices_per_geom), dtype=int)
            vertices = np.asarray(model.mesh_vert[start + pick], dtype=float)
            samples.append(vertices @ wxyz_to_rotation(model.geom_quat[g]).T + model.geom_pos[g])
        if not samples:
            raise RuntimeError(f"No collision-mesh vertex samples for {finger}")
        out[finger] = np.concatenate(samples)
    return out


class NewtonPatchDiagnostic:
    """One-environment, debug-only bridge from *actual* MJWarp events to patches.

    Uses the same solver diagnostic internals as validate_fingertip_contacts.py.
    Never writes contact tensors, physics state, scene objects or observations.
    Its ``contact_pad`` has OR-over-current-active-constraints semantics.
    """

    def __init__(self, probe, env, *, cfg: PatchSelection = _DEFAULT_SELECTION, show_patches: bool = False):
        self.probe = probe
        self.cfg = cfg
        self.show_patches = show_patches
        self.patches = candidate_patches_from_compiled_mjcf(
            probe.solver.mj_model, probe.geometry_bodies, probe.tip_bodies, cfg)
        self.link_indices = {}
        for finger, link in DISTAL_LINKS.items():
            found = [i for i, body in enumerate(env.scene["robot"].body_names)
                     if body.split("/")[-1] == link]
            if len(found) != 1:
                raise RuntimeError(f"Expected one articulated body for {link}, got {found}")
            self.link_indices[finger] = found[0]
        self.collision_vertex_samples = sampled_collision_mesh_vertices(
            probe.solver.mj_model, probe.geometry_bodies, probe.tip_bodies)
        self.last_contact_pad = np.zeros(5, dtype=np.int32)
        self.last_points = []
        self._markers = None
        self._show_geometry_report()
        self._report_mesh_equivalence()
        if show_patches:
            self._setup_visualization()
            self.sample(env)

    def _show_geometry_report(self):
        import json
        print("PATCH_MODEL_SOURCE compiled MuJoCo collision meshes from current Newton MJWarp model")
        print(f"PATCH_EQUIVALENT_ORDINARY_AABB {equivalent_ordinary_collision_bounds(self.patches)} ")
        for finger in FINGERS:
            patch = self.patches[finger]
            print("PATCH_DEFINITION " + json.dumps({
                "finger": finger, "link": patch.link,
                "collision_geom_ids": patch.collision_geom_ids,
                "collision_bounds_m": patch.collision_bounds_m,
                "center_local_m": patch.center_local_m.tolist(),
                "dimensions_m": patch.dimensions_m.tolist(),
                "orientation_local_wxyz": patch.orientation_local_wxyz,
                "normal_local": patch.normal_local,
                "inclusion_tolerance_m": patch.inclusion_tolerance_m,
                "max_contact_distance_m": self.cfg.max_signed_distance_m,
                "qualification": patch.rationale,
            }), flush=True)

    def _report_mesh_equivalence(self):
        """Compare all body-frame vertices AND indexed faces, not only bounds."""
        model = self.probe.solver.mj_model
        def meshes(finger):
            result = []
            for g in self.patches[finger].collision_geom_ids:
                m = int(model.geom_dataid[g])
                start, count = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
                vertices = np.asarray(model.mesh_vert[start:start+count])
                vertices = vertices @ wxyz_to_rotation(model.geom_quat[g]).T + model.geom_pos[g]
                start, count = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
                result.append((vertices, np.asarray(model.mesh_face[start:start+count])))
            return result
        reference = meshes("index")
        for finger in ("middle", "ring", "little"):
            other = meshes(finger)
            equal = len(reference) == len(other) and all(
                a.shape == b.shape and np.allclose(a, b, atol=1e-7, rtol=0)
                and np.array_equal(fa, fb)
                for (a, fa), (b, fb) in zip(reference, other, strict=True))
            print(f"PATCH_FULL_MESH_EQUIVALENCE index={finger} vertices_and_faces={equal}", flush=True)

    def _setup_visualization(self):
        """All markers are purely visual. No additional collisions are authored."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        greens = {}
        for finger, patch in self.patches.items():
            # Use separate markers to accommodate the thumb dimensions
            greens[finger] = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path=f"/Visuals/HX5CandidatePatch/{finger}",
                markers={"candidate": sim_utils.CuboidCfg(
                    size=tuple(float(x) for x in patch.dimensions_m),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0.85, 0.15)),
                )},
            ))
        current = {}
        for kind, colour in (("inside", (1., 0.87, 0.)), ("outside", (1., 0.1, 0.1))):
            current[kind] = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path=f"/Visuals/HX5CurrentContact/{kind}",
                markers={"current": sim_utils.SphereCfg(
                    radius=.0028, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour),
                )},
            ))
            current[kind].set_visibility(False)
        vertices = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/HX5CompiledCollisionVertexSamples",
            markers={"collision_vertex": sim_utils.SphereCfg(
                radius=.0009,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.62, .65, .70)),
            )},
        ))
        self._markers = (greens, current, vertices)
        print("PATCH_VIEWER green=proxy patch; yellow=current inside; red=current outside. ")
        print("PATCH_VIEWER grey dots=sampled compiled collision mesh vertices; not a triangulated mesh or tactile sensor.")

    @staticmethod
    def _poses(env):
        value = env.scene["robot"].data.body_link_pose_w
        return (value.torch if hasattr(value, "torch") else value)[0].detach().cpu().numpy()

    def _render(self, env, poses, events):
        import torch
        greens, points_markers, vertices = self._markers
        mesh_world = []
        for finger, local in self.collision_vertex_samples.items():
            pose = poses[self.link_indices[finger]]
            rot = xyzw_to_rotation(pose[3:7])
            mesh_world.extend((local @ rot.T + pose[:3]).tolist())
        vertices.visualize(translations=torch.tensor(mesh_world, dtype=torch.float32, device=env.device))
        for finger, marker in greens.items():
            pose = poses[self.link_indices[finger]]
            xyz = local_to_world(self.patches[finger].center_local_m, pose[:3], pose[3:7])
            marker.visualize(translations=torch.tensor(np.asarray([xyz]), dtype=torch.float32, device=env.device),
                             orientations=torch.tensor(np.asarray([pose[3:7]]), dtype=torch.float32, device=env.device))
        for kind, marker in points_markers.items():
            xyz = [ev["world_m"] for ev in events if ev["inside"] == (kind == "inside")]
            if not xyz:
                marker.set_visibility(False)  # current-state visibility, never stale history
            else:
                marker.set_visibility(True)
                marker.visualize(translations=torch.tensor(np.asarray(xyz), dtype=torch.float32, device=env.device))

    def sample(self, env):
        """Classify current active cube–distal constraints without latching earlier contacts."""
        data = self.probe.solver.mjw_data
        n = int(data.nacon.numpy()[0])  # only one environment is supported
        pairs = data.contact.geom.numpy()[:n]
        addresses = data.contact.efc_address.numpy()[:n]
        positions = data.contact.pos.numpy()[:n]
        signed_distances = data.contact.dist.numpy()[:n]
        poses = self._poses(env)
        bits = np.zeros(5, dtype=np.int32)
        events = []
        for pair, address, world, signed_distance in zip(
            pairs, addresses, positions, signed_distances, strict=True
        ):
            pair = np.asarray(pair, dtype=int)
            # Exclude inactive constraints and unused geometry slots in the solver buffer
            if pair.shape != (2,) or np.any(pair < 0) or not np.any(np.asarray(address) >= 0):
                continue
            if not (0 <= pair[0] < len(self.probe.geometry_bodies)
                    and 0 <= pair[1] < len(self.probe.geometry_bodies)):
                continue
            body1, body2 = (int(self.probe.geometry_bodies[g]) for g in pair)
            if (body1 in self.probe.object_bodies) == (body2 in self.probe.object_bodies):
                continue  # require exactly ONE cube side
            for i, finger in enumerate(FINGERS):
                if (body1 in self.probe.tip_bodies[finger]) + (body2 in self.probe.tip_bodies[finger]) != 1:
                    continue
                pose = poses[self.link_indices[finger]]
                accepted, local = active_contact_is_inside(
                    self.patches[finger], world, pose[:3], pose[3:7],
                    actual_pair=True, active_constraint=True,
                    signed_distance_m=float(signed_distance),
                    max_signed_distance_m=self.cfg.max_signed_distance_m,
                )
                bits[i] |= int(accepted)
                events.append({
                    "finger": finger, "geom_pair": pair.tolist(),
                    "world_m": [float(x) for x in world],
                    "local_m": [float(x) for x in local],
                    "signed_distance_m": float(signed_distance),
                    "inside": accepted,
                })
        self.last_contact_pad = bits
        self.last_points = events
        if self._markers is not None:
            self._render(env, poses, events)
        return bits.copy(), events


# ---------------------------------------------------------------------------
# Production privileged labels. Same patch geometry and event criteria as
# NewtonPatchDiagnostic, but classified in batches on the simulation device.
# ---------------------------------------------------------------------------


def classify_patch_contacts(
    *,
    num_envs: int,
    contact_geom: torch.Tensor,
    contact_addresses: torch.Tensor,
    contact_pos_w: torch.Tensor,
    contact_dist: torch.Tensor,
    contact_world: torch.Tensor,
    num_active: torch.Tensor,
    geom_tags: torch.Tensor,
    body_poses_w: torch.Tensor,
    finger_link_ids: torch.Tensor,
    lower_local: torch.Tensor,
    upper_local: torch.Tensor,
    tolerance_m: torch.Tensor,
    max_signed_distance_m: float,
) -> torch.Tensor:
    """One current-step binary cube/patch contact per (environment, finger).

    The compiled geom map assigns each collision geom a tag: finger 0..4,
    cube 5, or another body -1. A label requires an active cube-to-distal
    constraint, distance <= configured limit, and its contact point inside the
    corresponding candidate patch in the *live* distal link frame.
    """

    if contact_geom.ndim != 2 or contact_geom.shape[-1] != 2:
        raise ValueError("Expected contact.geom with shape [capacity, 2]")
    capacity = contact_geom.shape[0]
    if contact_addresses.ndim != 2 or contact_addresses.shape[0] != capacity:
        raise ValueError("Expected one constraint-address row per contact")
    if contact_pos_w.shape != (capacity, 3) or contact_dist.shape != (capacity,):
        raise ValueError("Invalid contact position or distance shape")
    if contact_world.shape != (capacity,) or num_active.numel() != 1:
        raise ValueError("Invalid contact world IDs or active count")
    if body_poses_w.ndim != 3 or body_poses_w.shape[0] != num_envs or body_poses_w.shape[-1] != 7:
        raise ValueError("Expected live body poses [envs, links, xyz+xyzw]")
    if geom_tags.ndim != 1 or geom_tags.numel() == 0 or finger_link_ids.shape != (5,):
        raise ValueError("Invalid compiled geom mapping or fingertip link IDs")
    if lower_local.shape != (5, 3) or upper_local.shape != (5, 3) or tolerance_m.shape != (5,):
        raise ValueError("Expected five patch boxes and tolerances")

    # Avoid contact.nacon.cpu(), .item() and NumPy copies at each physics step.
    # The capacity is fixed, while a GPU mask excludes unused solver entries.
    valid_pairs = ((contact_geom >= 0) & (contact_geom < geom_tags.numel())).all(dim=-1)
    pairs = contact_geom.long().clamp(0, geom_tags.numel() - 1)
    a, b = geom_tags[pairs[:, 0]], geom_tags[pairs[:, 1]]
    a_is_tip = (a >= 0) & (a < 5) & (b == 5)
    b_is_tip = (b >= 0) & (b < 5) & (a == 5)
    finger = torch.where(a_is_tip, a, torch.where(b_is_tip, b, -1))
    finger_id = finger.clamp(0, 4)
    world_id = contact_world.long().clamp(0, num_envs - 1)

    # Body poses are Isaac Lab XYZW quaternions; inverse-rotate each world
    # contact point into the matching distal body frame in one GPU operation.
    poses = body_poses_w[world_id, finger_link_ids[finger_id]]
    displacement = contact_pos_w - poses[:, :3]
    q_vec, q_w = poses[:, 3:6], poses[:, 6:7]
    cross = torch.linalg.cross(q_vec, displacement, dim=-1)
    local = displacement - 2 * q_w * cross + 2 * torch.linalg.cross(q_vec, cross, dim=-1)
    inside = (
        (local >= lower_local[finger_id] - tolerance_m[finger_id, None])
        & (local <= upper_local[finger_id] + tolerance_m[finger_id, None])
    ).all(dim=-1)

    accepted = (
        (torch.arange(capacity, device=contact_geom.device) < num_active.reshape(-1)[0])
        & valid_pairs
        & (finger >= 0)
        & (contact_world >= 0)
        & (contact_world < num_envs)
        & (contact_addresses >= 0).any(dim=-1)
        & torch.isfinite(contact_pos_w).all(dim=-1)
        & torch.isfinite(contact_dist)
        & (contact_dist <= max_signed_distance_m)
        & inside
    )
    # OR over any current active constraints in each environment/finger pair.
    bits = torch.zeros(num_envs * 5, device=contact_geom.device, dtype=torch.float32)
    bits.scatter_reduce_(0, world_id * 5 + finger_id, accepted.float(), reduce="amax", include_self=True)
    return bits.reshape(num_envs, 5)


def _to_torch(value) -> torch.Tensor:
    """Read Isaac Lab proxy arrays or native Warp buffers without CPU copies."""
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    if hasattr(value, "warp"):
        value = value.warp
    import warp as wp

    return wp.to_torch(value)


class NewtonPatchLabels:
    """Cache compiled fingertip geometry once; read current Newton contacts."""

    def __init__(self, env, cfg: PatchSelection = _DEFAULT_SELECTION):
        from isaaclab_newton.physics import NewtonManager

        solver = NewtonManager._solver
        if solver is None or not hasattr(solver, "mjw_data"):
            raise RuntimeError("Geometric patch contact labels require Newton MJWarp")
        self.solver = solver
        self.num_envs = env.num_envs
        model = NewtonManager.get_model()
        labels = model.body_label
        mj_model = solver.mj_model

        # geom -> compiled MuJoCo body -> global Newton body; all environments
        # use the same fingertip collision-mesh bounds in their *local* frames.
        geom_bodies = solver.mjc_body_to_newton.numpy().reshape(-1)[mj_model.geom_bodyid]
        object_bodies = {i for i, name in enumerate(labels) if name.endswith("/Object")}
        finger_bodies = {
            finger: {i for i, name in enumerate(labels) if name.endswith("/" + DISTAL_LINKS[finger])}
            for finger in FINGERS
        }
        if not object_bodies or any(not bodies for bodies in finger_bodies.values()):
            raise RuntimeError("Missing Object or distal links in the compiled Newton model")

        # Derive each patch from one representative body's compiled collision
        # meshes; replicas have the same body-local mesh and need not all be read.
        representative = {finger: {min(ids)} for finger, ids in finger_bodies.items()}
        patches = candidate_patches_from_compiled_mjcf(mj_model, geom_bodies, representative, cfg)
        self.lower_local = torch.tensor([patches[f].lower_local_m for f in FINGERS], device=env.device, dtype=torch.float32)
        self.upper_local = torch.tensor([patches[f].upper_local_m for f in FINGERS], device=env.device, dtype=torch.float32)
        self.tolerance_m = torch.tensor([patches[f].inclusion_tolerance_m for f in FINGERS], device=env.device, dtype=torch.float32)
        self.max_signed_distance_m = cfg.max_signed_distance_m

        tags = np.full(mj_model.ngeom, -1, dtype=np.int64)
        for g, body in enumerate(geom_bodies):
            if int(body) in object_bodies:
                tags[g] = 5
            else:
                for index, finger in enumerate(FINGERS):
                    if int(body) in finger_bodies[finger]:
                        tags[g] = index
                        break
        self.geom_tags = torch.tensor(tags, device=env.device, dtype=torch.int64)
        indices = []
        for finger in FINGERS:
            matches = [i for i, name in enumerate(env.scene["robot"].body_names) if name.split("/")[-1] == DISTAL_LINKS[finger]]
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one {finger} distal link; found {matches}")
            indices.append(matches[0])
        self.finger_link_ids = torch.tensor(indices, device=env.device, dtype=torch.int64)

    def sample(self, env) -> torch.Tensor:
        contacts = self.solver.mjw_data.contact
        return classify_patch_contacts(
            num_envs=self.num_envs,
            contact_geom=_to_torch(contacts.geom),
            contact_addresses=_to_torch(contacts.efc_address),
            contact_pos_w=_to_torch(contacts.pos),
            contact_dist=_to_torch(contacts.dist),
            contact_world=_to_torch(contacts.worldid),
            num_active=_to_torch(self.solver.mjw_data.nacon),
            geom_tags=self.geom_tags,
            body_poses_w=_to_torch(env.scene["robot"].data.body_link_pose_w),
            finger_link_ids=self.finger_link_ids,
            lower_local=self.lower_local,
            upper_local=self.upper_local,
            tolerance_m=self.tolerance_m,
            max_signed_distance_m=self.max_signed_distance_m,
        )


def fingertip_patch_contact(env) -> torch.Tensor:
    """Privileged binary labels, shape (num_envs, 5); never policy inputs."""
    labeler = getattr(env, "_hx5_patch_labels", None)
    if labeler is None:
        labeler = NewtonPatchLabels(env)
        env._hx5_patch_labels = labeler
    return labeler.sample(env)
