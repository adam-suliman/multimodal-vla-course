"""URDF forward kinematics and bounded five-axis IK, independent of Isaac Sim."""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .config import JOINTS


def transform(xyz, rpy):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    t[:3, 3] = xyz
    return t


class Kinematics:
    def __init__(self, urdf, cfg):
        self.cfg = cfg
        self.joints = {}
        self.children = {}
        self.lower, self.upper = np.zeros(6), np.zeros(6)
        for j in ET.parse(urdf).getroot().findall("joint"):
            origin = j.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            child, parent = j.find("child").get("link"), j.find("parent").get("link")
            name = j.get("name")
            axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
            self.joints[child] = (parent, name, transform(xyz, rpy), axis)
            self.children[name] = child
            if name in JOINTS:
                i = JOINTS.index(name)
                self.lower[i], self.upper[i] = (
                    float(j.find("limit").get("lower")),
                    float(j.find("limit").get("upper")),
                )

    def link(self, q, name):
        if name == "base_link":
            return transform(self.cfg["robot"]["base_position"], [0, 0, 0])
        parent, joint, origin, axis = self.joints[name]
        motion = np.eye(4)
        if joint in JOINTS:
            motion[:3, :3] = Rotation.from_rotvec(
                axis * q[JOINTS.index(joint)]
            ).as_matrix()
        return self.link(q, parent) @ origin @ motion

    def tool(self, q):
        t = self.link(q, self.cfg["robot"]["tool_link"])
        return t @ transform(self.cfg["robot"]["tool_offset"], [0, 0, 0])

    def solve(self, target, initial, gripper, direction=(0, 0, -1)):
        target = np.asarray(target)
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)

        def residual(x):
            t = self.tool(np.r_[x, gripper])
            # Position plus tool approach axis: SO-101 has five arm DOFs.
            return np.r_[
                t[:3, 3] - target,
                0.07 * (t[:3, 2] - direction),
                0.002 * (x - initial[:5]),
            ]

        rng = np.random.default_rng(0)
        best = None
        for guess in [
            np.array(initial[:5]),
            *rng.uniform(self.lower[:5], self.upper[:5], (12, 5)),
        ]:
            result = least_squares(
                residual,
                np.clip(guess, self.lower[:5] + 1e-6, self.upper[:5] - 1e-6),
                bounds=(self.lower[:5], self.upper[:5]),
                max_nfev=150,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
            if (
                np.linalg.norm(result.fun[:3]) < 0.001
                and np.linalg.norm(result.fun[3:6]) < 0.004
            ):
                break
        q = np.r_[best.x, gripper]
        error = np.linalg.norm(self.tool(q)[:3, 3] - target)
        if (
            error > self.cfg["expert"]["ik_tolerance_m"]
            or np.linalg.norm(best.fun[3:6]) > 0.015
        ):
            raise RuntimeError(
                f"Unreachable expert waypoint {target.tolist()}; IK error {error:.4f} m"
            )
        return q.astype(np.float32)
