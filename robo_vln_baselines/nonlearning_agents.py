import json
from collections import defaultdict

import numpy as np
from habitat import Env, logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm
from robo_vln_baselines.common.continuous_path_follower import (
    ContinuousPathFollower,
    track_waypoint
    
)
import shutil
import os
import random
from fastdtw import fastdtw
import habitat_sim
import gzip
from robo_vln_baselines.common.env_utils import construct_env
import wandb
from habitat.utils.visualizations import maps

from habitat.utils.visualizations.utils import (
    append_text_to_image,
    images_to_video,
)

def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )

from habitat.utils.visualizations.utils import observations_to_image

def save_map(observations, info, images):

    im = observations_to_image(observations,info )
    # im = observations["rgb"]
    top_down_map = draw_top_down_map(
        info, im.shape[0]
    )

    # depth_obs = observations['depth'].squeeze(2)
    # depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="RGB")
    # output_im = np.concatenate((im, top_down_map), axis=1)
    output_im = im
    output_im = append_text_to_image(
        output_im, observations["instruction"]["text"]
    )
    images.append(output_im)


def euclidean_distance(position_a, position_b):
    return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

def evaluate_agent(config: Config):
    split = config.EVAL.SPLIT
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.freeze()
    logger.info(config)

    env = construct_env(config)

    gt_path = config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=config.TASK_CONFIG.DATASET.SPLIT)
    with gzip.open(gt_path, "rt") as f:
        gt_json = json.load(f)

    assert config.EVAL.NONLEARNING.AGENT in [
        "RandomAgent",
        "HandcraftedAgent",
    ], "EVAL.NONLEARNING.AGENT must be either RandomAgent or HandcraftedAgent."

    if config.EVAL.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomContinuousAgent()
    else:
        agent = HandcraftedAgent()
    obs = env.reset()
    agent.reset()
    steps =0
    is_done = False
    stats_episodes = {}  # dict of dicts that stores stats per episode
    ep_count =0
    locations=[]

    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.controlling_ang_vel = True
    vel_control.ang_vel_is_local = True
    images = []
    IMAGE_DIR = os.path.join("examples", "images")
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    while (len(stats_episodes) < config.EVAL.EPISODE_COUNT):
        current_episode = env.habitat_env.current_episode
        actions = agent.act()
        vel_control.linear_velocity = np.array([0, 0, -actions[0]])
        vel_control.angular_velocity = np.array([0, actions[1], 0])
        observations, _, done, info = env.step(vel_control)
        episode_over, success = done
        episode_success = success and (actions[0]<0.25)
        is_done = episode_over or episode_success 
        steps+=1
        locations.append(env.habitat_env._sim.get_agent_state().position.tolist())
        save_map(observations, info, images)

        dirname = os.path.join(
            IMAGE_DIR, "icra_video", "%02d" % env.habitat_env.current_episode.episode_id
        )
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)

        if is_done or steps==config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS:
            gt_locations = gt_json[str(current_episode.episode_id)]["locations"]
            dtw_distance = fastdtw(locations, gt_locations, dist=euclidean_distance)[0]
            nDTW = np.exp(-dtw_distance / (len(gt_locations) * config.TASK_CONFIG.TASK.NDTW.SUCCESS_DISTANCE))

            locations=[]
            is_done = False
            ep_count+=1
            steps=0
            print("dones:", done)
            stats_episodes[current_episode.episode_id] = info
            stats_episodes[current_episode.episode_id]['ndtw'] = nDTW
            print("len stats episodes",len(stats_episodes))
            print("Current episode ID:", current_episode.episode_id)
            print("Episode Completed:", ep_count)
            print(" Episode done---------------------------------------------")
            obs = env.reset()
            print(stats_episodes[current_episode.episode_id])
            time_step = 1.0/30
            images_to_video(images, dirname, str(current_episode.episode_id), fps = int (1.0/time_step))
            images = []

    env.close()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    # print("-----------------------------------------------")
    # print(stats_episodes.values())
    for stat_key in next(iter(stats_episodes.values())).keys():
        # for v in stats_episodes.values():
        #     print (stat_key, "-------", v[stat_key])
        aggregated_stats[stat_key] = (
            sum([v[stat_key] for v in stats_episodes.values()]) / num_episodes
        )
    with open(f"stats_complete_{config.EVAL.NONLEARNING.AGENT}_{split}.json", "w") as f:
        json.dump(aggregated_stats, f, indent=4)


class RandomContinuousAgent(Agent):
    r"""Selects an action at each time step by sampling from the oracle action
    distribution of the training set.
    """

    def __init__(self):
        self.vel =0
        self.omega=0

    def reset(self):
        pass

    def act(self):
        self.vel =random.random() * 2.0
        self.omega= (random.random() - 0.5) * 2.0
        return (self.vel,self.omega)


class HandcraftedAgentContinuous(Agent):
    r"""Agent picks a random heading and takes 37 forward actions (average
    oracle path length) before calling stop.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 9.27m avg oracle path length in Train.
        # Fwd step size: 0.25m. 9.25m/0.25m = 37
        self.forward_steps = 30
        self.turns = np.random.randint(0, int(360 / 15) + 1)

    def act(self, observations):
        if self.turns > 0:
            self.turns -= 1
            return {"action": HabitatSimActions.TURN_RIGHT}
        if self.forward_steps > 0:
            self.forward_steps -= 1
            return {"action": HabitatSimActions.MOVE_FORWARD}
        return {"action": HabitatSimActions.STOP}

class HandcraftedAgent(Agent):
    r"""Agent picks a random heading and takes 37 forward actions (average
    oracle path length) before calling stop.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 9.27m avg oracle path length in Train.
        # Fwd step size: 0.25m. 9.25m/0.25m = 37
        self.forward_steps = 37
        self.turns = np.random.randint(0, int(360 / 15) + 1)

    def act(self, observations):
        if self.turns > 0:
            self.turns -= 1
            return {"action": HabitatSimActions.TURN_RIGHT}
        if self.forward_steps > 0:
            self.forward_steps -= 1
            return {"action": HabitatSimActions.MOVE_FORWARD}
        return {"action": HabitatSimActions.STOP}
