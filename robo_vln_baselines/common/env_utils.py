import random
from typing import Type, Union

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from habitat_baselines.common.env_utils import make_env_fn
from habitat.core.logging import logger

from robo_vln_baselines.common.environments import VLNCEDaggerEnv

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def construct_env(
    config: Config
) -> Env:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created.
        auto_reset_done: Whether or not to automatically reset the env on done

    Returns:
        VectorEnv object created according to specification.
    """

    num_processes = config.NUM_PROCESSES
    configs = []
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_processes:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

        random.shuffle(scenes)

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)
    
    for i in range(num_processes):
        new_config = config.clone()
        task_config = new_config.TASK_CONFIG.clone()
        task_config.defrost()
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]
        
        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID[i % len(config.SIMULATOR_GPU_ID)]
        )

        logger.info(
            f"Simulator GPU ID {config.SIMULATOR_GPU_ID}")

        logger.info(
            f"Simulator GPU ID {task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID}")
        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS
        task_config.freeze()

        new_config.defrost()
        new_config.TASK_CONFIG = task_config
        new_config.freeze()
        configs.append(new_config)


    # for i in range(num_processes):
    #     proc_config = config.clone()
    #     proc_config.defrost()

    #     task_config = proc_config.TASK_CONFIG
    #     if len(scenes) > 0:
    #         task_config.DATASET.CONTENT_SCENES = scene_splits[i]

    #     task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
    #         config.SIMULATOR_GPU_ID[i % len(config.SIMULATOR_GPU_ID)]
    #     )
    #     # task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_ID

    #     task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

    #     proc_config.freeze()
    #     configs.append(proc_config)

    for config in configs:
        logger.info(
            f"[construct_envs] Using GPU ID {config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID}")
    env = VLNCEDaggerEnv(config)
    return env

        
def construct_envs(
    config: Config, env_class: Type[Union[Env, RLEnv]], auto_reset_done: bool = True
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created.
        auto_reset_done: Whether or not to automatically reset the env on done

    Returns:
        VectorEnv object created according to specification.
    """

    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_processes:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

        random.shuffle(scenes)

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)
    
    for i in range(num_processes):
        new_config = config.clone()
        task_config = new_config.TASK_CONFIG.clone()
        task_config.defrost()
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID[i % len(config.SIMULATOR_GPU_ID)]
        )
        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS
        task_config.freeze()

        new_config.defrost()
        new_config.TASK_CONFIG = task_config
        new_config.freeze()
        configs.append(new_config)


    # for i in range(num_processes):
    #     proc_config = config.clone()
    #     proc_config.defrost()

    #     task_config = proc_config.TASK_CONFIG
    #     if len(scenes) > 0:
    #         task_config.DATASET.CONTENT_SCENES = scene_splits[i]

    #     task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
    #         config.SIMULATOR_GPU_ID[i % len(config.SIMULATOR_GPU_ID)]
    #     )
    #     # task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_ID

    #     task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

    #     proc_config.freeze()
    #     configs.append(proc_config)

    for config in configs:
        logger.info(
            f"[construct_envs] Using GPU ID {config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID}")
    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(tuple(zip(configs, env_classes))),
        auto_reset_done=auto_reset_done,
    )
    return envs


def construct_envs_auto_reset_false(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> VectorEnv:
    return construct_envs(config, env_class, auto_reset_done=False)
