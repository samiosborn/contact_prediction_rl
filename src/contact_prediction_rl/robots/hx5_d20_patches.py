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
    ordinary_x_tail: float = 0.38
    ordinary_z_tail: float = 0.38
    ordinary_y_central: float = 0.84
    thumb_y_tail: float = 0.35
    thumb_x_central: float = 0.80
    thumb_z_central: float = 0.80
    inclusion_tolerance_m: float = 0.0015
    max_signed_distance_m: float = 0.002

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
