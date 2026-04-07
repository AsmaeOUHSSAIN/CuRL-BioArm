"""Shape-tracing environment wrapper for the MyoSuite myoArm reach task.

Generates a sequence of 3D waypoints describing a target shape , 
augments the base observation with waypoint information,
and shapes the reward to encourage tracing the path in order.
"""

import numpy as np
from gymnasium import Wrapper, spaces

from constants import DEFAULT_CENTER_Y, CENTER_X, CENTER_Z


def generate_star_3d_path(center, outer_radius, inner_radius, n_branches=5, depth=0.05):
    cx, cy, cz = center
    half_d = depth / 2.0
    total_vertices = n_branches * 2

    front, back = [], []
    for i in range(total_vertices):
        angle = (2 * np.pi * i) / total_vertices - np.pi / 2
        r = outer_radius if i % 2 == 0 else inner_radius
        x = cx + r * np.cos(angle)
        z = cz + r * np.sin(angle)
        front.append(np.array([x, cy + half_d, z]))
        back.append(np.array([x, cy - half_d, z]))

    path = []
    for i in range(total_vertices):
        path.append(front[i])
    path.append(front[0])
    path.append(back[0])
    for i in range(1, total_vertices):
        path.append(back[i])
    path.append(back[0])

    on_back_side = True
    for i in range(1, total_vertices):
        if on_back_side:
            path.append(back[i])
            path.append(front[i])
            on_back_side = False
        else:
            path.append(front[i])
            path.append(back[i])
            on_back_side = True

    return {"path": path, "front": front, "back": back, "n_real_edges": 3 * total_vertices}


def generate_square_path(center, half_side, rotation_y_deg=0.0, points_per_edge=5):
    cx, cy, cz = center
    s = half_side
    angle = np.radians(rotation_y_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    corners_local = [
        np.array([-s, 0, -s]),
        np.array([ s, 0, -s]),
        np.array([ s, 0,  s]),
        np.array([-s, 0,  s]),
    ]

    corners_world = []
    for p in corners_local:
        rotated = np.array([
            cos_a * p[0] + sin_a * p[2],
            p[1],
            -sin_a * p[0] + cos_a * p[2],
        ])
        corners_world.append(np.array([cx, cy, cz]) + rotated)

    path = []
    n_corners = len(corners_world)
    for i in range(n_corners):
        start = corners_world[i]
        end = corners_world[(i + 1) % n_corners]
        for j in range(points_per_edge):
            t = j / points_per_edge
            path.append(start + t * (end - start))
    path.append(corners_world[0])

    return {"path": path, "corners": corners_world, "n_real_edges": 4}


class ShapeTracingWrapper(Wrapper):
    """Wraps a myoArm reach env into a shape-tracing task.

    The wrapper places a sequence of waypoints in 3D space, exposes them
    as part of the observation, and rewards the agent for visiting each
    waypoint in order within a tolerance.
    """

    def __init__(
        self,
        env,
        shape="star",
        center=None,
        scale=0.08,
        tolerance=0.02,
        depth=0.04,
        n_branches=5,
        inner_ratio=0.4,
        rotation_y=0.0,
        bonus_wp=1.0,
        bonus_complete=50.0,
        points_per_edge=5,
        deviation_penalty=0.0,
    ):
        super().__init__(env)

        self.shape_type = shape
        self.scale = scale
        self.tolerance = tolerance
        self.depth = depth
        self.n_branches = n_branches
        self.inner_ratio = inner_ratio
        self.rotation_y = rotation_y
        self.bonus_wp = bonus_wp
        self.bonus_complete = bonus_complete
        self.points_per_edge = points_per_edge
        self.deviation_penalty = deviation_penalty

        if center is None:
            self.center = np.array([CENTER_X, DEFAULT_CENTER_Y, CENTER_Z])
        else:
            self.center = np.array(center)

        self._generate_waypoints()

        self.current_wp_idx = 0
        self.waypoints_reached = 0
        self.cumulative_tracking_error = 0.0
        self.cumulative_edge_deviation = 0.0
        self.distance_traveled = 0.0
        self._prev_fingertip_pos = None
        self.steps_in_episode = 0
        self.reached_mask = np.zeros(self.n_waypoints, dtype=np.float32)

        orig_obs_space = self.observation_space
        n_wp = self.n_waypoints
        low = np.concatenate([
            orig_obs_space.low,
            np.full(n_wp * 3, -5.0),
            np.zeros(n_wp),
            [0.0],
        ])
        high = np.concatenate([
            orig_obs_space.high,
            np.full(n_wp * 3, 5.0),
            np.ones(n_wp),
            [1.0],
        ])
        self.observation_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
        )

    def _generate_waypoints(self):
        if self.shape_type == "star":
            result = generate_star_3d_path(
                center=self.center,
                outer_radius=self.scale,
                inner_radius=self.scale * self.inner_ratio,
                n_branches=self.n_branches,
                depth=self.depth,
            )
        elif self.shape_type == "square":
            result = generate_square_path(
                center=self.center,
                half_side=self.scale,
                rotation_y_deg=self.rotation_y,
                points_per_edge=self.points_per_edge,
            )
        else:
            raise ValueError(f"Unknown shape '{self.shape_type}'. Use 'star' or 'square'.")

        self.waypoints = result["path"]
        self.n_waypoints = len(self.waypoints)

    def set_scale(self, scale):
        self.scale = scale
        self._generate_waypoints()

    def set_tolerance(self, tol):
        self.tolerance = tol

    def set_rotation_y(self, angle_deg):
        self.rotation_y = angle_deg
        self._generate_waypoints()

    def set_center_y(self, y):
        self.center[1] = y
        self._generate_waypoints()

    def _get_fingertip_pos(self):
        return self.unwrapped.sim.data.site_xpos[self.unwrapped.tip_sids[0]].copy()

    def _set_target_position(self, pos):
        self.unwrapped.sim.model.site_pos[self.unwrapped.target_sids[0]] = pos.copy()
        self.unwrapped.sim.forward()

    @staticmethod
    def _point_to_segment_dist(point, seg_start, seg_end):
        edge = seg_end - seg_start
        edge_len_sq = np.dot(edge, edge)
        if edge_len_sq < 1e-12:
            return np.linalg.norm(point - seg_start)
        t = np.clip(np.dot(point - seg_start, edge) / edge_len_sq, 0.0, 1.0)
        projection = seg_start + t * edge
        return np.linalg.norm(point - projection)

    def _augment_obs(self, obs):
        wp_positions = np.concatenate(self.waypoints).astype(np.float32)
        progress = self.current_wp_idx / max(self.n_waypoints, 1)
        return np.concatenate([obs, wp_positions, self.reached_mask, [progress]]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self._generate_waypoints()
        self.current_wp_idx = 0
        self.waypoints_reached = 0
        self.cumulative_tracking_error = 0.0
        self.cumulative_edge_deviation = 0.0
        self.distance_traveled = 0.0
        self._prev_fingertip_pos = None
        self.steps_in_episode = 0
        self.reached_mask = np.zeros(self.n_waypoints, dtype=np.float32)

        self._set_target_position(self.waypoints[0])
        self.unwrapped.robot.sync_sims(self.unwrapped.sim, self.unwrapped.sim_obsd)
        obs = self.unwrapped.get_obs_vec()

        info["shape_type"] = self.shape_type
        info["n_waypoints"] = self.n_waypoints
        info["scale"] = self.scale
        info["tolerance"] = self.tolerance
        return self._augment_obs(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.steps_in_episode += 1

        fingertip_pos = self._get_fingertip_pos()
        if self._prev_fingertip_pos is not None:
            self.distance_traveled += np.linalg.norm(fingertip_pos - self._prev_fingertip_pos)
        self._prev_fingertip_pos = fingertip_pos.copy()

        target_pos = self.waypoints[self.current_wp_idx]
        dist = np.linalg.norm(fingertip_pos - target_pos)
        self.cumulative_tracking_error += dist

        seg_start = self.waypoints[self.current_wp_idx - 1] if self.current_wp_idx > 0 else self.waypoints[0]
        edge_deviation = self._point_to_segment_dist(fingertip_pos, seg_start, target_pos)
        self.cumulative_edge_deviation += edge_deviation

        reward = -dist - self.deviation_penalty * edge_deviation

        wp_reached_this_step = False
        if dist < self.tolerance:
            self.reached_mask[self.current_wp_idx] = 1.0
            self.current_wp_idx += 1
            self.waypoints_reached += 1
            wp_reached_this_step = True
            reward += self.bonus_wp

            if self.current_wp_idx >= self.n_waypoints:
                terminated = True
                reward += self.bonus_complete
                info["shape_completed"] = True
                info["success"] = True
            else:
                self._set_target_position(self.waypoints[self.current_wp_idx])
                self.unwrapped.robot.sync_sims(self.unwrapped.sim, self.unwrapped.sim_obsd)

        terminated = terminated and (
            info.get("shape_completed", False) or self.current_wp_idx >= self.n_waypoints
        )

        info["current_wp_idx"] = self.current_wp_idx
        info["n_waypoints"] = self.n_waypoints
        info["waypoint_progress"] = self.current_wp_idx / self.n_waypoints
        info["current_distance"] = dist
        info["waypoints_reached"] = self.waypoints_reached
        info["mean_tracking_error"] = self.cumulative_tracking_error / self.steps_in_episode
        info["wp_reached_this_step"] = wp_reached_this_step
        info["scale"] = self.scale
        info["distance_traveled"] = self.distance_traveled
        info["edge_deviation"] = edge_deviation
        info["mean_edge_deviation"] = self.cumulative_edge_deviation / self.steps_in_episode

        obs = self.unwrapped.get_obs_vec()
        return self._augment_obs(obs), reward, terminated, truncated, info
