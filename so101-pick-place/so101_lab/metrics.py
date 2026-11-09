"""Geometry-based evaluator. Privileged state never enters policy observations."""

from dataclasses import dataclass
from itertools import product
import numpy as np
from .config import PROTOCOL


def cube_corners(position, rotation, size):
    local = np.array(list(product((-size / 2, size / 2), repeat=3)))
    return local @ np.asarray(rotation).T + np.asarray(position)


@dataclass
class SuccessTracker:
    initial_z: float
    target_xy: tuple
    support_z: float
    cube_size: float = 0.03
    target_size: tuple = (0.1, 0.1)
    lifted: bool = False
    stable_time: float = 0.0
    success: bool = False
    max_lift: float = 0.0
    bilateral_contact_seen: bool = False

    def update(
        self,
        *,
        position,
        rotation,
        linear_velocity,
        angular_velocity,
        finger_contacts,
        gripper_open,
        dt,
    ):
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Evaluator time step must be finite and positive.")
        arrays = [position, rotation, linear_velocity, angular_velocity]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Non-finite privileged simulator state.")
        self.max_lift = max(self.max_lift, float(position[2] - self.initial_z))
        self.lifted |= self.max_lift >= PROTOCOL["lift_m"]
        self.bilateral_contact_seen |= all(finger_contacts)
        corners = cube_corners(position, rotation, self.cube_size)
        inside = np.all(
            np.abs(corners[:, :2] - self.target_xy) <= np.array(self.target_size) / 2
        )
        # Resting on the table rules out an airborne cube and holding above the target.
        supported = (
            abs(corners[:, 2].min() - self.support_z) <= PROTOCOL["support_tolerance_m"]
        )
        slow = (
            np.linalg.norm(linear_velocity) <= PROTOCOL["linear_speed_m_s"]
            and np.linalg.norm(angular_velocity) <= PROTOCOL["angular_speed_rad_s"]
        )
        released = bool(gripper_open) and not any(finger_contacts)
        valid = self.lifted and inside and supported and slow and released
        self.stable_time = self.stable_time + dt if valid else 0.0
        self.success |= self.stable_time + 1e-9 >= PROTOCOL["stable_s"]
        return self.success

    def result(self):
        return {
            "success": bool(self.success),
            "lifted": bool(self.lifted),
            "max_lift_m": self.max_lift,
            "stable_time_s": self.stable_time,
            "bilateral_contact_seen": bool(self.bilateral_contact_seen),
        }
