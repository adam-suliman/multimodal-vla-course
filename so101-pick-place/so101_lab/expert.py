"""Privileged Cartesian pick-and-place expert; all motion uses joint drives."""

import numpy as np


class Expert:
    def __init__(self, kinematics, cfg):
        self.ik, self.cfg = kinematics, cfg

    def reset(self, cube_position, state):
        c = self.cfg
        source = np.array(cube_position, dtype=float)
        source[2] += c["expert"]["grasp_z_offset"]
        target = np.r_[
            c["target"]["center_xy"], source[2] + c["expert"]["place_clearance"]
        ]
        above_source, above_target = source.copy(), target.copy()
        above_source[2] += c["expert"]["clearance"]
        above_target[2] = above_source[2]
        opened, closed = c["robot"]["open_rad"], c["robot"]["closed_rad"]
        self.stages = [
            ("approach", above_source, opened, 3.0),
            ("descend", source, opened, 2.5),
            ("grasp", source, closed, 1.5),
            ("lift", above_source, closed, 2.5),
            ("transfer", above_target, closed, 3.5),
            ("lower", target, closed, 2.5),
            ("release", target, opened, 1.5),
            ("retreat", above_target, opened, 2.5),
            ("settle", above_target, opened, 3.0),
        ]
        self.index, self.elapsed = 0, 0.0
        self.start_position = self.ik.tool(state)[:3, 3]
        self.start_axis = self.ik.tool(state)[:3, 2]
        self.start_gripper = float(state[5])
        self.last = np.array(state)

    @property
    def stage(self):
        return self.stages[self.index][0]

    def action(self, observation):
        _, goal, grip, duration = self.stages[self.index]
        increment = 1 / self.cfg["control_hz"]
        max_step = self.cfg["robot"]["max_speed_rad_s"] * increment
        # Slow Cartesian progress at difficult IK branches instead of jumping joints.
        for _ in range(10):
            fraction = min((self.elapsed + increment) / duration, 1.0)
            smooth = fraction * fraction * (3 - 2 * fraction)
            xyz = (1 - smooth) * self.start_position + smooth * goal
            g = (1 - smooth) * self.start_gripper + smooth * grip
            direction = (1 - smooth) * self.start_axis + smooth * np.array(
                [0.0, 0.0, -1.0]
            )
            candidate = self.ik.solve(xyz, self.last, g, direction)
            if np.max(np.abs(candidate - self.last)) <= max_step + 1e-6:
                break
            increment *= 0.5
        else:
            raise RuntimeError(
                "Expert IK could not produce a continuous joint trajectory."
            )
        self.elapsed += increment
        self.last = candidate
        if fraction >= 1 and self.index < len(self.stages) - 1:
            self.index += 1
            self.elapsed = 0.0
            self.start_position, self.start_gripper = goal.copy(), grip
            self.start_axis = np.array([0.0, 0.0, -1.0])
        return self.last.copy()
