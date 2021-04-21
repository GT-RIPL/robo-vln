import random
import quaternion
import habitat_sim

import matplotlib.pyplot as plt
import json
import numpy as np
from habitat_sim.utils.common import quat_to_magnum
import magnum as mn

import habitat

class ContinuousPathFollower(object):
    def __init__(self, sim, path, waypoint_threshold):
        self._sim = sim
        self._points = np.array(path[:])
        assert len(self._points) > 0
        self._length = self._sim.geodesic_distance(path[0], path[-1])
        self._threshold = waypoint_threshold
        self._step_size = 0.01
        self.progress = 0  # geodesic distance -> [0,1]
        self.waypoint = np.array(path[0])

        # setup progress waypoints
        _point_progress = [0]
        _segment_tangents = []
        _length = self._length
        for ix, point in enumerate(self._points):
            if ix > 0:
                segment = point - self._points[ix - 1]
                segment_length = np.linalg.norm(segment)
                segment_tangent = segment / segment_length
                _point_progress.append(
                    segment_length / _length + _point_progress[ix - 1]
                )
                # t-1 -> t
                _segment_tangents.append(segment_tangent)
        self._point_progress = _point_progress
        self._segment_tangents = _segment_tangents
        # final tangent is duplicated
        self._segment_tangents.append(self._segment_tangents[-1])

        # print("self._length = " + str(self._length))
        # print("num points = " + str(len(self._points)))
        # print("self._point_progress = " + str(self._point_progress))
        # print("self._segment_tangents = " + str(self._segment_tangents))

    def pos_at(self, progress):
        if progress <= 0:
            return self._points[0]
        elif progress >= 1.0:
            return self._points[-1]

        path_ix = 0
        for ix, prog in enumerate(self._point_progress):
            if prog > progress:
                path_ix = ix
                break

        segment_distance = self._length * (progress - self._point_progress[path_ix - 1])
        return (
            self._points[path_ix - 1]
            + self._segment_tangents[path_ix - 1] * segment_distance
        )

    def update_waypoint(self):
        if self.progress < 1.0:
            wp_disp = self.waypoint - self._sim.get_agent_state().position
            wp_dist = np.linalg.norm(wp_disp)
            node_pos = self._sim.get_agent_state().position
            step_size = self._step_size
            threshold = self._threshold
            while wp_dist < threshold:
                self.progress += step_size
                self.waypoint = self.pos_at(self.progress)
                if self.progress >= 1.0:
                    break
                wp_disp = self.waypoint - node_pos
                wp_dist = np.linalg.norm(wp_disp)


def setup_path_visualization(sim, path_follower, vis_samples=100):
    vis_ids = []
    sphere_handle = obj_attr_mgr.get_template_handles("uvSphereSolid")[0]
    sphere_template_cpy = obj_attr_mgr.get_template_by_handle(sphere_handle)
    sphere_template_cpy.scale *= 0.2
    template_id = obj_attr_mgr.register_template(sphere_template_cpy, "mini-sphere")
    print("template_id = " + str(template_id))
    if template_id < 0:
        return None
    vis_ids.append(sim.add_object_by_handle(sphere_handle))

    for point in path_follower._points:
        cp_id = sim.add_object_by_handle(sphere_handle)
        if cp_id < 0:
            print(cp_id)
            return None
        sim.set_translation(point, cp_id)
        vis_ids.append(cp_id)

    for i in range(vis_samples):
        cp_id = sim.add_object_by_handle("mini-sphere")
        if cp_id < 0:
            print(cp_id)
            return None
        sim.set_translation(path_follower.pos_at(float(i / vis_samples)), cp_id)
        vis_ids.append(cp_id)

    for id in vis_ids:
        if id < 0:
            print(id)
            return None

    for id in vis_ids:
        sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, id)

    return vis_ids

def remove_all_objects(sim):
    for id in sim.get_existing_object_ids():
        sim.remove_object(id)


def track_waypoint(waypoint, rs, vc, progress, dt=1.0 / 30.0):
    angular_error_threshold = 0.5
    max_linear_speed = 1.0
    max_turn_speed = 1.0
    glob_forward = rs.rotation.transform_vector(mn.Vector3(0, 0, -1.0)).normalized()
    glob_right = rs.rotation.transform_vector(mn.Vector3(-1.0, 0, 0)).normalized()
    to_waypoint = mn.Vector3(waypoint) - rs.translation
    u_to_waypoint = to_waypoint.normalized()
    angle_error = float(mn.math.angle(glob_forward, u_to_waypoint))

    new_velocity = 0
    if progress >0.985:
         # slow down to 0
        new_velocity = (vc.linear_velocity[2]) / 1.5
    elif angle_error < angular_error_threshold:
        # speed up to max
        new_velocity = (vc.linear_velocity[2] - max_linear_speed) / 2.0
    else:
        # slow down to 0
        new_velocity = (vc.linear_velocity[2]) / 2.0
    vc.linear_velocity = mn.Vector3(0, 0, new_velocity)

    # angular part
    rot_dir = 1.0
    if mn.math.dot(glob_right, u_to_waypoint) < 0:
        rot_dir = -1.0
    angular_correction = 0.0
    if angle_error > (max_turn_speed * 10.0 * dt):
        angular_correction = max_turn_speed
    else:
        angular_correction = angle_error / 2.0

    vc.angular_velocity = mn.Vector3(
        0, np.clip(rot_dir * angular_correction, -max_turn_speed, max_turn_speed), 0
    )
    return new_velocity, np.clip(rot_dir * angular_correction, -max_turn_speed, max_turn_speed)

# def track_waypoint(waypoint, rs, vc, dt=1.0 / 60.0):
#     angular_error_threshold = 0.5
#     max_linear_speed = 1.0
#     max_turn_speed = 1.0
#     glob_forward = rs.rotation.transform_vector(mn.Vector3(0, 0, -1.0)).normalized()
#     glob_right = rs.rotation.transform_vector(mn.Vector3(-1.0, 0, 0)).normalized()
#     to_waypoint = mn.Vector3(waypoint) - rs.translation
#     u_to_waypoint = to_waypoint.normalized()
#     angle_error = float(mn.math.angle(glob_forward, u_to_waypoint))

#     new_velocity = 0
#     if angle_error < angular_error_threshold:
#         # speed up to max
#         new_velocity = (vc.linear_velocity[2] - max_linear_speed) / 2.0
#     else:
#         # slow down to 0
#         new_velocity = (vc.linear_velocity[2]) / 2.0
#     vc.linear_velocity = mn.Vector3(0, 0, new_velocity)

#     # angular part
#     rot_dir = 1.0
#     if mn.math.dot(glob_right, u_to_waypoint) < 0:
#         rot_dir = -1.0
#     angular_correction = 0.0
#     if angle_error > (max_turn_speed * 10.0 * dt):
#         angular_correction = max_turn_speed
#     else:
#         angular_correction = angle_error / 2.0

#     vc.angular_velocity = mn.Vector3(
#         0, np.clip(rot_dir * angular_correction, -max_turn_speed, max_turn_speed), 0
#     )
#     return new_velocity, np.clip(rot_dir * angular_correction, -max_turn_speed, max_turn_speed)
