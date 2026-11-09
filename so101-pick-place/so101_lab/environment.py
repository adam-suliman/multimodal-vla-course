"""Isaac Sim 5.1 environment. Import only after constructing SimulationApp."""

import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .assets import fetch, robot_urdf, sha256, prepare_import_urdf
from .config import ROOT, JOINTS, PROTOCOL
from .kinematics import Kinematics
from .metrics import SuccessTracker


def camera_quaternion(position, target):
    back = np.asarray(position, float) - target
    back /= np.linalg.norm(back)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(up, back)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return Rotation.from_matrix(np.column_stack((right, up, back))).as_quat()[
        [3, 0, 1, 2]
    ]


def convert_robot(data_dir):
    """Import pinned URDF with fixed base and convex decomposition, without GPU physics."""
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Usd, UsdPhysics, UsdGeom

    lock = fetch(data_dir, ("robot",), offline=True)
    target = Path(data_dir).resolve() / "converted/so101.usd"
    meta = target.with_suffix(".json")
    recipe = {
        "revision": lock["robot"]["revision"],
        "isaac": lock["isaac_image"],
        "simulator_lock_sha256": sha256(ROOT / "requirements-simulator.lock"),
        "recipe_version": 2,
        "fixed_base": True,
        "remove_empty_tcp_rigid_body": True,
        "convex_decomposition": True,
        "merge_fixed_joints": True,
        "self_collision": False,
    }
    if target.exists() and meta.exists():
        recorded = json.loads(meta.read_text())
        if recorded["recipe"] == recipe and all(
            (target.parent / name).is_file() and sha256(target.parent / name) == digest
            for name, digest in recorded["files"].items()
        ):
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    source = prepare_import_urdf(data_dir, target.parent / "urdf/so101.urdf")
    enable_extension("isaacsim.asset.importer.urdf")
    ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not ok:
        raise RuntimeError("Could not initialize the Isaac Sim URDF importer.")
    config.set_fix_base(True)
    config.set_merge_fixed_joints(True)
    config.set_self_collision(False)
    config.set_convex_decomp(True)
    config.set_create_physics_scene(False)
    config.set_make_default_prim(True)
    config.set_distance_scale(1.0)
    config.set_default_drive_strength(40.0)
    config.set_default_position_drive_damping(3.0)
    ok, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(source),
        import_config=config,
        dest_path=str(target),
    )
    if not ok or not target.exists():
        raise RuntimeError("URDF import failed; see the Isaac Sim log.")
    stage = Usd.Stage.Open(str(target))
    if not stage or not stage.GetDefaultPrim():
        raise RuntimeError("Converted robot has no default prim.")
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    revolute = [p.GetName() for p in prims if p.IsA(UsdPhysics.RevoluteJoint)]
    meshes = sum(p.IsA(UsdGeom.Mesh) for p in prims)
    collisions = sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims)
    if (
        set(revolute) != set(JOINTS)
        or len(revolute) != 6
        or meshes < 1
        or collisions < 1
    ):
        raise RuntimeError(
            "Converted asset is missing joints, visual meshes or collision shapes."
        )
    files = {
        str(p.relative_to(target.parent)): sha256(p)
        for p in target.parent.rglob("*")
        if p.is_file() and p != meta
    }
    meta.write_text(
        json.dumps(
            {
                "recipe": recipe,
                "files": files,
                "meshes": meshes,
                "collision_shapes": collisions,
            },
            indent=2,
        )
        + "\n"
    )
    return target


class Environment:
    """reset(seed), observe(), step(joint_targets), result(), close().

    Public observations contain exactly two RGB images, joint radians and task text.
    Physics advances at a fixed rate and pauses during policy inference.
    """

    def __init__(self, cfg, data_dir):
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleArticulation, RigidPrim, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera
        from pxr import UsdLux, UsdPhysics, PhysxSchema, Gf, UsdShade

        self.cfg = cfg
        self.ik = Kinematics(robot_urdf(data_dir), cfg)
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1 / cfg["physics_hz"],
            rendering_dt=1 / cfg["control_hz"],
            backend="numpy",
            device="cpu",
        )
        physics = self.world.get_physics_context()
        physics.enable_gpu_dynamics(False)
        physics.set_broadphase_type("MBP")
        # Publish poses to USD for this small, single-environment CPU scene.
        physics.enable_fabric(False)
        stage = self.world.stage
        material = PhysicsMaterial(
            "/World/contact_material",
            static_friction=cfg["contact"]["static_friction"],
            dynamic_friction=cfg["contact"]["dynamic_friction"],
            restitution=0.0,
        )
        self.support_z = cfg["table"]["center"][2] + cfg["table"]["size"][2] / 2
        self.world.scene.add(
            FixedCuboid(
                "/World/table",
                name="table",
                size=1,
                position=np.array(cfg["table"]["center"]),
                scale=np.array(cfg["table"]["size"]),
                color=np.array(cfg["table"]["color"]),
                physics_material=material,
            )
        )
        # The target is a visual marking; its top is flush with the support surface.
        self.world.scene.add(
            VisualCuboid(
                "/World/target",
                name="target",
                size=1,
                position=np.r_[cfg["target"]["center_xy"], self.support_z + 0.0001],
                scale=np.r_[cfg["target"]["size"], 0.0002],
                color=np.array(cfg["target"]["color"]),
            )
        )
        self.cube = self.world.scene.add(
            DynamicCuboid(
                "/World/cube",
                name="cube",
                size=cfg["cube"]["size"],
                position=np.r_[
                    cfg["cube"]["xy_min"], self.support_z + cfg["cube"]["size"] / 2
                ],
                color=np.array(cfg["cube"]["color"]),
                mass=cfg["cube"]["mass"],
                physics_material=material,
            )
        )
        asset = convert_robot(data_dir)
        add_reference_to_stage(str(asset), "/World/robot")
        SingleXFormPrim(
            "/World/robot", translation=np.array(cfg["robot"]["base_position"])
        )
        # Imported meshes are instanceable. Make this single robot editable so
        # contact offsets and friction are applied to its actual collision shapes.
        while True:
            instances = [
                p
                for p in stage.Traverse()
                if str(p.GetPath()).startswith("/World/robot") and p.IsInstance()
            ]
            if not instances:
                break
            for prim in instances:
                prim.SetInstanceable(False)
        # Override only the printed plastic in the composed scene. Keep the
        # checksum-verified source, converted asset and dark servos unchanged.
        printed_shader = UsdShade.Shader(
            stage.GetPrimAtPath("/World/robot/Looks/material_a_d_printed/Shader")
        )
        printed_color = printed_shader.GetInput("diffuse_color_constant")
        if not printed_color:
            raise RuntimeError(
                "The locked SO-101 asset has no printed-plastic color input."
            )
        printed_color.Set(Gf.Vec3f(*cfg["robot"]["printed_color"]))
        links = {}
        articulation_path = None
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/World/robot"):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    links[prim.GetName()] = str(prim.GetPath())
                    PhysxSchema.PhysxRigidBodyAPI.Apply(
                        prim
                    ).CreateSolverPositionIterationCountAttr(16)
                    PhysxSchema.PhysxRigidBodyAPI.Apply(
                        prim
                    ).CreateSolverVelocityIterationCountAttr(4)
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    articulation_path = str(prim.GetPath())
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    # The importer defaults to mass-normalized acceleration
                    # drives. Our gains and effort limits specify torque drives.
                    UsdPhysics.DriveAPI(prim, "angular").GetTypeAttr().Set("force")
                if prim.IsA(UsdPhysics.FixedJoint):
                    joint = UsdPhysics.FixedJoint(prim)
                    if not joint.GetBody0Rel().GetTargets():
                        joint.GetLocalPos0Attr().Set(
                            Gf.Vec3f(*cfg["robot"]["base_position"])
                        )
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr(cfg["contact"]["contact_offset"])
                api.CreateRestOffsetAttr(cfg["contact"]["rest_offset"])
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material.material, materialPurpose="physics"
                )
        if articulation_path is None:
            raise RuntimeError("Converted URDF has no articulation root.")
        self.robot = self.world.scene.add(
            SingleArticulation(articulation_path, name="robot")
        )
        fingers = [links["gripper_link"], links["moving_jaw_so101_v1_link"]]
        self.contacts = self.world.scene.add(
            RigidPrim(
                "/World/cube",
                name="cube_contacts",
                track_contact_forces=True,
                contact_filter_prim_paths_expr=fingers,
                max_contact_count=64,
                reset_xform_properties=False,
            )
        )
        light = UsdLux.DomeLight.Define(stage, "/World/light")
        light.CreateIntensityAttr(1000.0)
        self.cameras = {}
        for name in ("top", "wrist"):
            c = cfg["cameras"][name]
            parent = "/World" if name == "top" else links[c["parent_link"]]
            cam = Camera(
                f"{parent}/{name}_camera",
                name=f"{name}_camera",
                resolution=tuple(cfg["cameras"]["resolution"]),
            )
            cam.set_local_pose(
                translation=np.array(c["position"]),
                orientation=camera_quaternion(c["position"], c["look_at"]),
                camera_axes="usd",
            )
            # Scale both optical lengths equally to avoid RTX's small-focal-length
            # clamp. Pinhole projection depends only on their ratio; DOF is off.
            cam.set_horizontal_aperture(100 * cfg["cameras"]["horizontal_aperture_m"])
            cam.set_focal_length(100 * cfg["cameras"]["focal_length_m"])
            cam.set_clipping_range(0.005, 10.0)
            self.cameras[name] = cam
        self.world.reset()
        self.indices = np.array([self.robot.get_dof_index(name) for name in JOINTS])
        self._configure_drives()
        for cam in self.cameras.values():
            cam.initialize()
            cam.set_lens_aperture(0.0)
        self.tracker = None
        self.ticks = 0
        self.clipped_actions = 0

    def _configure_drives(self):
        if self.robot.num_dof != 6:
            raise RuntimeError(f"Expected six SO-101 DOFs, found {self.robot.num_dof}")
        kps, kds = np.empty(6), np.empty(6)
        kps[self.indices] = self.cfg["robot"]["stiffness"]
        kds[self.indices] = self.cfg["robot"]["damping"]
        self.robot.get_articulation_controller().set_gains(
            kps=kps, kds=kds, save_to_usd=True
        )
        self.robot.get_articulation_controller().set_max_efforts(
            np.array(self.cfg["robot"]["effort_limit"]), joint_indices=self.indices
        )

    def reset(self, seed):
        from isaacsim.core.utils.types import ArticulationAction

        self.world.reset()
        self._configure_drives()
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        xy = rng.uniform(self.cfg["cube"]["xy_min"], self.cfg["cube"]["xy_max"])
        home = np.array(self.cfg["robot"]["home_rad"], dtype=np.float32)
        self.robot.set_joint_positions(home, joint_indices=self.indices)
        self.robot.set_joint_velocities(np.zeros(6), joint_indices=self.indices)
        self.robot.apply_action(
            ArticulationAction(joint_positions=home, joint_indices=self.indices)
        )
        self.last_action = home.copy()
        self.cube.set_world_pose(
            np.r_[xy, self.support_z + self.cfg["cube"]["size"] / 2],
            np.array([1.0, 0, 0, 0]),
        )
        self.cube.set_linear_velocity(np.zeros(3))
        self.cube.set_angular_velocity(np.zeros(3))
        for _ in range(self.cfg["physics_hz"]):
            self.world.step(render=False, update_fabric=True)
        self._capture()
        self.initial_position = self.cube.get_world_pose()[0].copy()
        if np.linalg.norm(self.initial_position[:2] - xy) > 0.002:
            raise RuntimeError(
                "Cube moved during reset settling; inspect the home pose."
            )
        self.tracker = SuccessTracker(
            float(self.initial_position[2]),
            tuple(self.cfg["target"]["center_xy"]),
            self.support_z,
            self.cfg["cube"]["size"],
            tuple(self.cfg["target"]["size"]),
        )
        self.ticks, self.clipped_actions = 0, 0
        return self.observe()

    def observe(self):
        images = {}
        width, height = self.cfg["cameras"]["resolution"]
        for name, cam in self.cameras.items():
            rgba = cam.get_rgba()
            if rgba is None or rgba.shape != (height, width, 4):
                raise RuntimeError(f"Camera {name} did not produce an RGBA frame.")
            images[name] = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
        return {
            "images": images,
            "state": self.robot.get_joint_positions(joint_indices=self.indices).astype(
                np.float32
            ),
            "task": self.cfg["task"],
        }

    def _capture(self, check_time=False):
        import carb
        import omni.replicator.core as rep

        # Synchronous RGB capture without advancing unaccounted physics time.
        before = self.world.current_time
        self.world.render()  # Flush the most recent physics poses to Fabric.
        settings = carb.settings.get_settings()
        enabled = settings.get("/app/player/playSimulations")
        settings.set("/app/player/playSimulations", False)
        try:
            # Warm materials after reset; assets stay unchanged within episodes.
            rep.orchestrator.step(
                rt_subframes=1 if check_time else 4,
                delta_time=0.0,
                pause_timeline=False,
            )
        finally:
            settings.set("/app/player/playSimulations", enabled)
        if check_time and abs(self.world.current_time - before) > 1e-7:
            raise RuntimeError("Camera capture advanced unrecorded physics time.")

    def privileged_state(self):
        position, quat = self.cube.get_world_pose()
        force = np.asarray(
            self.contacts.get_contact_force_matrix(dt=1 / self.cfg["physics_hz"])
        )
        touches = (np.linalg.norm(force.reshape(2, 3), axis=1) > 1e-4).tolist()
        return {
            "position": position,
            "rotation": Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix(),
            "linear_velocity": self.cube.get_linear_velocity(),
            "angular_velocity": self.cube.get_angular_velocity(),
            "finger_contacts": touches,
            "gripper_open": bool(
                self.robot.get_joint_positions(joint_indices=self.indices)[5] > 0.6
            ),
        }

    def step(self, joint_targets):
        from isaacsim.core.utils.types import ArticulationAction

        if self.tracker is None or self.done:
            raise RuntimeError("Reset the environment before stepping a new episode.")
        action = np.asarray(joint_targets, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError(
                "Action must contain six finite absolute joint targets in radians."
            )
        delta = self.cfg["robot"]["max_speed_rad_s"] / self.cfg["control_hz"]
        action_safe = np.clip(
            np.clip(action, self.ik.lower, self.ik.upper),
            self.last_action - delta,
            self.last_action + delta,
        ).astype(np.float32)
        self.clipped_actions += int(not np.allclose(action, action_safe, atol=1e-6))
        self.last_action = action_safe.copy()
        self.robot.apply_action(
            ArticulationAction(joint_positions=action_safe, joint_indices=self.indices)
        )
        self.last_samples = []
        for _ in range(self.cfg["physics_hz"] // self.cfg["control_hz"]):
            before = self.world.current_time
            self.world.step(render=False, update_fabric=True)
            if (
                abs(self.world.current_time - before - 1 / self.cfg["physics_hz"])
                > 1e-7
            ):
                raise RuntimeError(
                    "Actual simulator time step differs from the protocol."
                )
            self.ticks += 1
            sample = self.privileged_state()
            self.tracker.update(**sample, dt=1 / self.cfg["physics_hz"])
            self.last_samples.append(
                {
                    **{
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in sample.items()
                    },
                    "tick": self.ticks,
                    "success": bool(self.tracker.success),
                }
            )
        self._capture(check_time=True)
        return self.observe(), self.done

    @property
    def done(self):
        return self.tracker.success or self.ticks >= round(
            PROTOCOL["time_limit_s"] * self.cfg["physics_hz"]
        )

    def result(self):
        return {
            **self.tracker.result(),
            "seed": self.seed,
            "sim_time_s": self.ticks / self.cfg["physics_hz"],
            "initial_position": self.initial_position.tolist(),
            "clipped_actions": self.clipped_actions,
        }

    def close(self):
        self.world.stop()
        self.world.clear_instance()
