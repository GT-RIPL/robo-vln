from typing import Any

import numpy as np
import cv2
import habitat
from gym import spaces
from PIL import Image
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# from habitat_extensions.sensor_utils import (
#     constrain_to_pm_pi,
#     gen_point_cloud,
#     safe_assign,
#     scale_vertical_points,
#     to_grid,
#     get_extrinsics
# )
# from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector



@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    r"""The agents current location in the global coordinate frame

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "globalgps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        return agent_position.astype(np.float32)


@registry.register_sensor
class VLNOracleActionSensor(Sensor):
    r"""Sensor for observing the optimal action to take. The assumption this
    sensor currently makes is that the shortest path to the goal is the
    optimal path.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)
        self.follower = ShortestPathFollower(
            self._sim,
            # all goals can be navigated to within 0.5m.
            goal_radius=getattr(config, "GOAL_RADIUS", 0.5),
            return_one_hot=False,
        )
        self.follower.mode = "geodesic_path"

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "vln_oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return np.array(
            [best_action if best_action is not None else HabitatSimActions.STOP]
        )


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    r"""Sensor for observing how much progress has been made towards the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "progress"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        # TODO: what is the correct sensor type?
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        distance_from_start = episode.info["geodesic_distance"]

        return (distance_from_start - distance_to_target) / distance_from_start

# @habitat.registry.register_sensor(name="map_sensor")
# class TopDownMapSensor(habitat.Sensor):
#     def __init__(self, sim, config, *args: Any, **kwargs: Any):
#         self.map_resolution = config.MAP_RESOLUTION
#         self.out_width = config.WIDTH
#         self.out_height = config.HEIGHT
#         self.colorize = config.COLORIZE
#         super().__init__(config=config, *args, **kwargs)

#         self._sim = sim
#         self.camera = None
#         self.extrinsic_matrix = None

#         self.max_depth = self.config.MAX_DEPTH
#         self.depth_threshold = (0.01, self.max_depth)
#         self.height_threshold = self.config.HEIGHT_THRESHOLD  # (0.2, 0.9)
#         self.grid_delta = 3

#         self.config_x_min, self.config_x_max = self.config.MAP_BOUNDS_X
#         self.config_y_min, self.config_y_max = self.config.MAP_BOUNDS_Y

#         self.map = np.zeros(self.map_resolution)

#     # Defines the name of the sensor in the sensor suite dictionary
#     def _get_uuid(self, *args: Any, **kwargs: Any):
#         return "map"

#     # Defines the type of the sensor
#     def _get_sensor_type(self, *args: Any, **kwargs: Any):
#         return habitat.SensorTypes.COLOR

#     # Defines the size and range of the observations of the sensor
#     def _get_observation_space(self, *args: Any, **kwargs: Any):
#         return spaces.Box(
#             low=0,
#             high=255,
#             shape=(self.out_height,
#                    self.out_width, 3),
#             dtype=np.uint8,
#         )

#     def reset(self):
#         self.map = np.zeros(self.map_resolution)

#         coordinates_min, coordinates_max = self.get_scene_size()
#         # self.coordinate_min = min(coordinates_min[2], coordinates_max[2])
#         # self.coordinate_max = max(coordinates_min[0], coordinates_max[0])
#         self.coordinate_min = min(coordinates_min)
#         self.coordinate_min = self.coordinate_min - \
#             0.5 * np.absolute(self.coordinate_min)
#         self.coordinate_max = max(coordinates_max) * 1.5

#     def get_scene_size(self):
#         curr_scene = self._sim._sim.semantic_scene
#         curr_scene_center = curr_scene.aabb.center
#         curr_scene_dims = np.absolute(curr_scene.aabb.sizes)

#         coordinates_min = curr_scene_center - curr_scene_dims / 2.
#         coordinates_max = curr_scene_center + curr_scene_dims / 2.

#         return coordinates_min, coordinates_max

#     @property
#     def palette(self):
#         '''
#         Set the default palette.
#         Where map is zero, color white; otherwise, color black.
#         '''
#         palette = np.zeros((256, 3), dtype=np.uint8)
#         palette[0] = np.array([255, 255, 255])

#         return palette

#     def colorize_map(self, map, resize=True):

#         if self.colorize and map.ndim == 2:
#             map_image = Image.new("P", (map.shape[1], map.shape[0]))
#             map_image.putpalette(self.palette.flatten())
#             map_image.putdata(map.flatten())
#             map_image = map_image.convert('RGB')
#         elif map.ndim == 3:
#             map_image = Image.fromarray(map)
#         else:
#             resize = False
#             map_image = map

#         if resize:
#             map_image = map_image.resize((self.out_width, self.out_height))

#         return np.array(map_image)

#     # This is called whenver reset is called or an action is taken
#     def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
#         self.camera = self._sim.get_agent_state()
#         extrinsic_matrix = self.get_extrinsic_from_camera(self.camera)
#         self._last_seen_area = self.get_seen_area(observations,
#                                                   extrinsic_matrix)
#         self.cropped_map = self.get_cropped_map()
#         #print('sum cropped map', np.sum(self.cropped_map>0))
#         self.color_map = self.colorize_map(self.cropped_map)

#         return self.color_map

#     @property
#     def last_seen_area(self):
#         return self._last_seen_area / self.map.size

#     @staticmethod
#     def get_extrinsic_from_camera(camera):
#         rotation_quaternion = camera.sensor_states['depth'].rotation
#         translation_matrix = camera.sensor_states['depth'].position
#         return get_extrinsics(translation_matrix, rotation_quaternion)

#     def to_grid(self, x, y):
#         return to_grid(x, y,
#                        coordinate_min=self.coordinate_min,
#                        coordinate_max=self.coordinate_max,
#                        grid_resolution=self.map_resolution)

#     def get_seen_area(self, observations, extrinsics):
#         depth = observations['depth']
#         rgb = observations['rgb']
#         points, points_colors = gen_point_cloud(depth, rgb, extrinsics,
#                                                 depth_threshold=self.depth_threshold)

#         if len(points):
#             points[points < self.coordinate_min] = self.coordinate_min
#             points[points > self.coordinate_max] = self.coordinate_max

#             grid_locs = self.to_grid(points[:, 0],
#                                      points[:, 2])
#             if grid_locs.max() > self.map_resolution[0]:
#                 assert points.min() > self.coordinate_min and points.max() < self.coordinate_max
#             grids_mat = np.zeros(self.map_resolution)
#             # normalize
#             points = scale_vertical_points(points)
#             high_filter_idx = points[:, 1] < self.height_threshold[1]
#             low_filter_idx = points[:, 1] > self.height_threshold[0]
#             obstacle_idx = np.logical_and(high_filter_idx, low_filter_idx)

#             safe_assign(grids_mat, grid_locs[high_filter_idx, 0],
#                         grid_locs[high_filter_idx, 1], 2)
#             kernel = np.ones((3, 3), np.uint8)
#             grids_mat = cv2.morphologyEx(grids_mat, cv2.MORPH_CLOSE, kernel)

#             obs_mat = np.zeros((grids_mat.shape[0], grids_mat.shape[1]),
#                                dtype=np.uint8)
#             safe_assign(obs_mat, grid_locs[obstacle_idx, 0],
#                         grid_locs[obstacle_idx, 1], 1)
#             kernel = np.ones((5, 5), np.uint8)
#             obs_mat = cv2.morphologyEx(obs_mat, cv2.MORPH_CLOSE, kernel)
#             obs_idx = np.where(obs_mat == 1)
#             safe_assign(grids_mat, obs_idx[0], obs_idx[1], 1)

#             self.map[np.where(grids_mat == 2)] = 2
#             self.map[np.where(grids_mat == 1)] = 1

#         seen_area = np.count_nonzero(self.map)
#         return seen_area

#     def get_cropped_map(self):
#         if self.config_x_min is None:
#             range_x = np.where(np.any(self.map, axis=1))[0]
#             range_y = np.where(np.any(self.map, axis=0))[0]
#             if range_x.size and range_y.size:
#                 self.ind_x_min = range_x[0]
#                 self.ind_x_max = range_x[-1]
#                 self.ind_y_min = range_y[0]
#                 self.ind_y_max = range_y[-1]
#             else:
#                 self.ind_x_min, self.ind_y_min = (0, 0)
#                 self.ind_x_max, self.ind_y_max = self.map.shape
#         else:
#             self.ind_x_min = self.config_x_min
#             self.ind_x_max = self.config_x_max
#             self.ind_y_min = self.config_y_min
#             self.ind_y_max = self.config_y_max

#         house_map = self.map[
#             max(self.ind_x_min - self.grid_delta, 0):
#             self.ind_x_max + self.grid_delta,
#             max(self.ind_y_min - self.grid_delta, 0):
#             self.ind_y_max + self.grid_delta,
#         ]
#         return house_map

# @habitat.registry.register_sensor(name="instance_map_sensor")
# class InstanceMapSensor(TopDownMapSensor):
#     # Defines the name of the sensor in the sensor suite dictionary
#     def _get_uuid(self, *args: Any, **kwargs: Any):
#         return "td_instance_map"

#     def get_seen_area(self, observations, extrinsics):
#         depth = observations['depth']
#         rgb = observations['semantic']
#         points, points_colors = gen_point_cloud(depth, rgb, extrinsics,
#                                                 depth_threshold=self.depth_threshold)

#         if len(points) > 1:
#             points_colors = np.squeeze(points_colors)
#             grid_locs = self.to_grid(points[:, 0],
#                                      points[:, 2])

#             grids_mat = np.zeros(self.map_resolution)
#             semantic_mat = np.zeros(self.map_resolution)
#             # normalize
#             points = scale_vertical_points(points)
#             high_filter_idx = points[:, 1] < self.height_threshold[1]
#             low_filter_idx = points[:, 1] > self.height_threshold[0]
#             obstacle_idx = np.logical_and(high_filter_idx, low_filter_idx)

#             safe_assign(grids_mat, grid_locs[high_filter_idx, 0],
#                         grid_locs[high_filter_idx, 1], 2)
#             safe_assign(semantic_mat, grid_locs[high_filter_idx, 0],
#                         grid_locs[high_filter_idx, 1],
#                         points_colors[high_filter_idx])
#             kernel = np.ones((3, 3), np.uint8)
#             grids_mat = cv2.morphologyEx(grids_mat, cv2.MORPH_CLOSE, kernel)

#             obs_mat = np.zeros((grids_mat.shape[0], grids_mat.shape[1]),
#                                dtype=np.uint8)
#             safe_assign(obs_mat, grid_locs[obstacle_idx, 0],
#                         grid_locs[obstacle_idx, 1], 1)
#             kernel = np.ones((5, 5), np.uint8)
#             obs_mat = cv2.morphologyEx(obs_mat, cv2.MORPH_CLOSE, kernel)
#             obs_idx = np.where(obs_mat == 1)
#             safe_assign(grids_mat, obs_idx[0], obs_idx[1], 1)

#             self.map[np.where(grids_mat != 0)] = np.int32((
#                 semantic_mat[np.where(grids_mat != 0)] % 40) + 1)
#         seen_area = np.sum(self.map > 0)
#         return seen_area

#     @property
#     def palette(self):
#         from habitat_sim.utils.common import d3_40_colors_rgb
#         palette = np.full((256, 3), 255, dtype=np.uint8)
#         palette[1:len(d3_40_colors_rgb) + 1] = d3_40_colors_rgb

#         return palette

# @habitat.registry.register_sensor(name="semantic_map_sensor")
# class SemanticMapSensor(InstanceMapSensor):
#     # Defines the name of the sensor in the sensor suite dictionary
#     def _get_uuid(self, *args: Any, **kwargs: Any):
#         return "td_semantic_map"

#     def reset(self):
#         super().reset()
#         curr_scene_objs = self._sim._sim.semantic_scene.objects
#         object2idx = [o.category.index() for o in curr_scene_objs]
#         self.objectname2idx = {o.category.name(): o.category.index()
#                                for o in curr_scene_objs}
#         self.object2idx = np.array(object2idx)

#     def get_seen_area(self, observations, extrinsics):

#         depth = observations['depth']
#         rgb = observations['semantic']
#         points, points_colors = gen_point_cloud(depth, rgb, extrinsics,
#                                                 depth_threshold=self.depth_threshold)

#         if len(points) > 1:
#             points_colors = np.squeeze(points_colors)
#             grid_locs = self.to_grid(points[:, 0],
#                                      points[:, 2])

#             grids_mat = np.zeros(self.map_resolution)
#             semantic_mat = np.zeros(self.map_resolution, dtype=np.uint8)

#             points = scale_vertical_points(points)
#             high_filter_idx = points[:, 1] < self.height_threshold[1]
#             low_filter_idx = points[:, 1] > self.height_threshold[0]
#             obstacle_idx = np.logical_and(high_filter_idx, low_filter_idx)

#             safe_assign(grids_mat, grid_locs[high_filter_idx, 0],
#                         grid_locs[high_filter_idx, 1], 2)
#             safe_assign(semantic_mat, grid_locs[high_filter_idx, 0],
#                         grid_locs[high_filter_idx, 1], points_colors[high_filter_idx])
#             kernel = np.ones((3, 3), np.uint8)
#             grids_mat = cv2.morphologyEx(grids_mat, cv2.MORPH_CLOSE, kernel)

#             obs_mat = np.zeros((grids_mat.shape[0], grids_mat.shape[1]),
#                                dtype=np.uint8)
#             safe_assign(obs_mat, grid_locs[obstacle_idx, 0],
#                         grid_locs[obstacle_idx, 1], 1)
#             kernel = np.ones((5, 5), np.uint8)
#             obs_mat = cv2.morphologyEx(obs_mat, cv2.MORPH_CLOSE, kernel)
#             obs_idx = np.where(obs_mat == 1)
#             safe_assign(grids_mat, obs_idx[0], obs_idx[1], 1)
#             try:
#                 self.map[np.where(grids_mat != 0)] = np.int32((
#                     self.object2idx[semantic_mat[np.where(grids_mat != 0)]] % 40) + 1)
#             except IndexError:
#                 # This occasionally throws an error, but not sure why yet
#                 shapes = [self.map.shape, grids_mat.shape, semantic_mat.shape]
#                 raise
#         seen_area = np.sum(self.map > 0)
#         return seen_area

# @habitat.registry.register_sensor(name="ego_map_sensor")        
# class EgoMapSensor(TopDownMapSensor):
#     def __init__(self, sim, config, *args: Any, **kwargs: Any):
#         self.map_range = config.MAP_RANGE
#         self.MAP_TYPE = config.MAP_TYPE
#         super().__init__(sim, config, *args, **kwargs)

#         map_sensor_type = registry.get_sensor(config.MAP_TYPE)

#         assert map_sensor_type is not None, "invalid sensor type {}".format(
#             config.MAP_TYPE
#         )
#         self.map_sensor = map_sensor_type(sim, config)

#     def _get_uuid(self, *args: Any, **kwargs: Any):
#         if "semantic" in self.MAP_TYPE:
#             uuid = "ego_sem_map"
#         elif "occupancy" in self.MAP_TYPE:
#             uuid = "ego_occ_map"

#         return uuid

#     def get_cropped_map(self):
#         return self.map

#     def reset(self):
#         super().reset()
#         self.map_sensor.reset()

#     def _get_observation_space(self, *args: Any, **kwargs: Any):
#         if self.colorize:
#             high = 255
#             shape = (self.out_height,
#                      self.out_width,
#                      3)
#         else:
#             shape = (self.map_range * 2, self.map_range * 2)
#             if "semantic" in self.MAP_TYPE:
#                 high = 40
#             elif "occupancy" in self.MAP_TYPE:
#                 high = 3

#         return spaces.Box(
#             low=0,
#             high=high,
#             shape=shape,
#             dtype=np.uint8,
#         )

#     def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
#         self.map_sensor.get_observation(observations, episode, *args, **kwargs)
#         self._last_seen_area = self.map_sensor._last_seen_area

#         top_down_map = self.map_sensor.map.copy()
#         loc_map = self.map_sensor.colorize_map(top_down_map, resize=False)
#         half_size = max(top_down_map.shape[0],
#                         top_down_map.shape[1],
#                         self.map_range) * 3
#         if loc_map.ndim > 2:
#             color = True
#             ego_map = np.ones((half_size * 2, half_size * 2, 3),
#                               dtype=np.uint8) * 255
#         else:
#             color = False
#             ego_map = np.zeros((half_size * 2, half_size * 2),
#                                dtype=loc_map.dtype)
#         current_pos, grid_current_pos = self.get_camera_grid_pos()
#         x_start = half_size - grid_current_pos[0]
#         y_start = half_size - grid_current_pos[1]
#         x_end = x_start + top_down_map.shape[0]
#         y_end = y_start + top_down_map.shape[1]
#         assert x_start >= 0 and y_start >= 0 and \
#             x_end <= ego_map.shape[0] and y_end <= ego_map.shape[1], \
#             "x_start, y_start: {}. x_end, y_end: {}. ego_map.shape: {}".format((x_start, y_start),
#                                                                                (x_end,
#                                                                                 y_end),
#                                                                                ego_map.shape)
#         ego_map[x_start: x_end, y_start: y_end] = loc_map
#         center = (half_size, half_size)
#         rot_angle = constrain_to_pm_pi(current_pos[2] - 90)
#         M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
#         ego_map = cv2.warpAffine(ego_map, M,
#                                  (ego_map.shape[1], ego_map.shape[0]),
#                                  flags=cv2.INTER_NEAREST,
#                                  borderMode=cv2.BORDER_CONSTANT,
#                                  borderValue=0)
#         start = half_size - self.map_range
#         end = half_size + self.map_range
#         small_ego_map = ego_map[start:end, start:end]

#         # if color:
#         #     output_map = np.array(Image.fromarray(
#         #         self.map_sensor.colorize_map(small_ego_map)).convert('RGBA'))
#         # else:
#         #     output_map = small_ego_map
#         # return output_map
#         return self.map_sensor.colorize_map(small_ego_map)

#     def get_camera_grid_pos(self):
#         camera_pos = self._sim.get_agent_state()
#         position = camera_pos.position
#         rotation = camera_pos.rotation
#         # theta, _ = quat_to_angle_axis(camera_pos.rotation)
#         # theta = np.rad2deg(theta)

#         direction_vector = np.array([0, 0, -1])

#         heading_vector = quaternion_rotate_vector(
#             rotation.inverse(), direction_vector
#         )

#         phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
#         theta = np.rad2deg(phi)

#         current_pos = [position[0],
#                        position[2],
#                        constrain_to_pm_pi(theta)]

#         grid_pos = self.to_grid([current_pos[0]], [current_pos[1]])
#         return current_pos, np.squeeze(grid_pos)

