import itertools
import math

import cv2
import numpy as np
import quaternion
from PIL import Image

from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import COORDINATE_MAX, COORDINATE_MIN
from habitat.utils.visualizations.utils import images_to_video


def print_scene_recur(scene, limit_output=10):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None


def img_to_world(depth, rgb, E, depth_threshold=None):
    H, W = depth.shape[:2]

    img_pixs = np.mgrid[0:H, 0:W].reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]  # swap (v, u) into (u, v)
    Img_pixs_ones = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))

    I = get_intrinsics(height=H, width=W, vfov=90)
    Iinv = np.linalg.inv(I[:3, :3])
    Cam_to_img_mat = np.dot(Iinv, Img_pixs_ones)  # have to match habitat

    depth_vals = depth.reshape(-1)
    depth_vals[np.isnan(
        depth_vals)] = 10.  # check nan and if nan set depth with max depth which is 10
    normalized_depth_vals = depth_vals / 10
    rgb_vals = rgb.reshape(len(depth_vals), -1)
    if depth_threshold is not None:
        # print('depth_th',depth_threshold)
        assert type(depth_threshold) is tuple
        valid = normalized_depth_vals >= depth_threshold[0]

        if len(depth_threshold) > 1:
            valid = np.logical_and(valid,
                                   normalized_depth_vals <= depth_threshold[1])
        depth_vals = depth_vals[valid]
        rgb_vals = rgb_vals[valid]
        points_in_cam = np.multiply(Cam_to_img_mat[:, valid], depth_vals)
    else:
        points_in_cam = np.multiply(Cam_to_img_mat, depth_vals)
    points_in_cam = np.concatenate((points_in_cam,
                                    np.ones((1, points_in_cam.shape[1]))),
                                   axis=0)
    points_in_world = np.dot(E, points_in_cam)
    points_in_world = points_in_world.T
    return points_in_world[:, :3], rgb_vals


def get_intrinsics(height, width, vfov):
    """
    calculate the intrinsic matrix from vertical_fov
    notice that hfov and vfov are different if height != width
    we can also get the intrinsic matrix from opengl's
    perspective matrix
    http://kgeorge.github.io/2014/03/08/calculating-opengl-
    perspective-matrix-from-opencv-intrinsic-matrix
    """
    vfov = vfov / 180.0 * np.pi
    tan_half_vfov = np.tan(vfov / 2.0)
    tan_half_hfov = tan_half_vfov * width / float(height)
    fx = width / 2.0 / tan_half_hfov  # focal length in pixel space
    fy = height / 2.0 / tan_half_vfov
    I = np.array([[fx, 0, width / 2.0, 0],
                  [0, fy, height / 2.0, 0],
                  [0, 0, 1, 0]])
    return I


def gen_point_cloud(depth, rgb, E, depth_threshold=(0.,), inv_E=False):

    if inv_E:
        E = np.linalg.inv(np.array(E).reshape((4, 4)))
    # different from real-world camera coordinate system
    # opengl uses negative z axis as the camera front direction
    # x axes are same, hence y axis is reversed as well
    # https://learnopengl.com/Getting-started/Camera
    rot = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    E = np.dot(E, rot)

    depth_vals = depth.reshape(-1)
    depth_vals[np.isnan(
        depth_vals)] = 10.  # check nan and if nan set depth with max depth which is 10
    normalized_depth_vals = depth_vals / 10

    points, points_rgb = img_to_world(depth, rgb, E,
                                      depth_threshold=depth_threshold)
    return points, points_rgb


def scale_vertical_points(points):
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    if np.abs((y_max - y_min)) > 1e-6:
        points[:, 1] = (points[:, 1] - y_min) / ((y_max - y_min) + 1e-6)
    else:
        points[:, 1] = np.zeros_like(points[:, 1])
    return points


# def to_grid(
#     realworld_x,
#     realworld_y,
#     coordinate_min=COORDINATE_MIN,
#     coordinate_max=COORDINATE_MAX,
#     grid_resolution=(1250, 1250),
# ):
#     """Return gridworld index of realworld coordinates assuming top-left corner
#     is the origin. The real world coordinates of lower left corner are
#     (coordinate_min, coordinate_min) and of top right corner are
#     (coordinate_max, coordinate_max)
#     """

#     grid_size = (
#         (coordinate_max - coordinate_min) / grid_resolution[0],
#         (coordinate_max - coordinate_min) / grid_resolution[1],
#     )
#     # grid_x = (coordinate_max - realworld_x) / grid_size[0]
#     # grid_y = (realworld_y - coordinate_min) / grid_size[1]
#     grid_x = (realworld_x - coordinate_min) / grid_size[0]
#     grid_y = (realworld_y - coordinate_min) / grid_size[0]
#     grid_xy = np.stack([grid_x, grid_y], axis=1)
#     return np.floor(grid_xy).astype(int)


def safe_assign(im_map, x_idx, y_idx, value):
    try:
        im_map[x_idx, y_idx] = value
    except IndexError:
        valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
        valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
        valid_idx = np.logical_and(valid_idx1, valid_idx2)
        im_map[x_idx[valid_idx], y_idx[valid_idx]] = value


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(info["top_down_map"]["map"],
                                             info["top_down_map"]["fog_of_war_mask"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def semantic_to_image(semantic_obs):
    from habitat_sim.utils.common import d3_40_colors_rgb
    semantic_img = Image.new(
        "P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")

    return np.array(semantic_img)


def get_extrinsics(translation_matrix, rotation_quaternion):
    rotation_matrix = quaternion.as_rotation_matrix(rotation_quaternion)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[0:3, 0:3] = rotation_matrix
    extrinsic_matrix[0:3, 3] = translation_matrix
    return extrinsic_matrix


def get_world_to_camera(translation_vector, rotation_quaternion):
    R = quaternion.as_rotation_matrix(rotation_quaternion)
    rotation_matrix = np.eye(4)
    rotation_matrix[0:3, 0:3] = R
    translation_matrix = np.eye(4)
    translation_matrix[0:3, 3] = -translation_vector
    extrinsic_matrix = rotation_matrix @ translation_matrix
    return extrinsic_matrix


def images_to_grid(imgs, grid_size):
    h, w = grid_size
    img_h, img_w, img_c = imgs[0].shape

    m_x = 0
    m_y = 0

    imgmatrix = np.zeros((img_h * h + m_y * (h - 1),
                          img_w * w + m_x * (w - 1),
                          img_c),
                         np.uint8)

    imgmatrix.fill(255)

    positions = itertools.product(range(w), range(h))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        if img.shape[-1] > img_c:
            img = img[..., :img_c]
        imgmatrix[y:y + img_h, x:x + img_w, :] = img

    return imgmatrix


def next_perfect_square(N):
    sqrt = math.sqrt(N)
    next_ = math.floor(sqrt)
    if next_ < sqrt:
        next_ += 1
    return next_


def get_grid_size(N):
    grid = (2, 3)
    if N < 4:
        grid = (1, 3)
    elif N > 6:
        square = next_perfect_square(N)
        grid = (square, square)

    return grid


def concatentate_and_write_video(observations,
                                 directory_to_save_video='./',
                                 video_filename='vid.mp4'):
    output_shape = observations[0][0].shape[:2]
    num_observations = len(observations[0])
    if num_observations == 1:
        output_all = [cv2.resize(obs[0], output_shape) for obs in observations]
    else:
        grid_size = get_grid_size(num_observations)
        observations = [[cv2.resize(o, output_shape)
                         for o in obs] for obs in observations]
        output_all = [images_to_grid(obs, grid_size)
                      for obs in observations]

    images_to_video(output_all,
                    directory_to_save_video,
                    video_filename)


def constrain_to_pm_pi(theta):
    # make sure theta is within [-180, 180)
    return (theta + 180) % 360 - 180
