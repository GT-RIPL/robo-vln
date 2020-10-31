import copy
import gc
import json
import os
import random
import time
import warnings
from collections import defaultdict
from typing import Dict
import matplotlib.pyplot as plt
import scipy.misc
import time
import habitat_sim
import gc
import time
import magnum as mn 
import time
import quaternion
from habitat_sim.utils.common import quat_to_magnum, quat_from_magnum
from fastdtw import fastdtw
import gzip 

import cv2

from PIL import Image
from habitat.utils.visualizations import maps

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
# WandB – Import the wandb library
import wandb
from habitat_baselines.common.utils import generate_video
from vlnce_baselines.common.continuous_path_follower import (
    ContinuousPathFollower,
    track_waypoint
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_env,
    construct_envs,
    construct_envs_auto_reset_false,
    SimpleRLEnv
)
from vlnce_baselines.common.utils import transform_obs, batch_obs, batch_obs_data_collect, repackage_hidden, split_batch_tbptt, repackage_mini_batch
from vlnce_baselines.models.cma_policy import CMANet
from vlnce_baselines.models.seq2seq_sem_attn import Seq2Seq_Sem_Attn_Policy
from vlnce_baselines.models.seq2seq_policy import Seq2SeqPolicy
from vlnce_baselines.models.seq2seq_text_attn import Seq2Seq_Lang_Attn
from vlnce_baselines.models.seq2seq_sem_text_attn import Seq2Seq_Sem_Text_Attn
from vlnce_baselines.models.hybrid_cma_mp import TransformerHybridPolicy
from vlnce_baselines.models.seq2seq import Seq2SeqNet

#WandB – Login to your wand
# b account so you can log all your metrics
wandb.login()

wandb.init(project="PM", sync_tensorboard=True)
# If you don't want your script to sync to the cloud
# os.environ['WANDB_MODE'] = 'dryrun'

wb_config = wandb.config
wb_config.LR = 1e-4
wb_config.EPOCHS = 20
wb_config.BATCH_SIZE = 1

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

# from pynvml import *
# nvmlInit()


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
            obs,
            prev_actions,
            oracle_actions,
            inflec_weight,
        )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t
        
        pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
        return torch.cat([t, pad], dim=0)
    
    def _pad_instruction(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(1)
        if pad_amount == 0:
            return t
        pad = torch.full_like(t[:,0], fill_val).expand(*t.size()[:1], pad_amount)
        return torch.cat([t, pad], dim=1)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    oracle_stop_batch = list(transposed[3])
    N = len(corrected_actions_batch)
#     weights_batch = list(transposed[3])
    B = len(prev_actions_batch)
    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        if sensor == 'instruction':
            for bid in range(N):
                new_observations_batch[sensor].append(observations_batch[bid][sensor])
        else: 
            for bid in range(B):
                new_observations_batch[sensor].append(observations_batch[bid][sensor])

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    max_insr_len = max(ele.size(1) for ele in observations_batch['instruction'])
    for bid in range(B):
        for sensor in observations_batch:
            if sensor == 'instruction':  
                observations_batch[sensor][bid] = _pad_instruction(
                    observations_batch[sensor][bid], max_insr_len, fill_val=0.0
                )
                continue
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=0.0
            )
        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len, fill_val=0.0
        )
        oracle_stop_batch[bid] = _pad_helper(oracle_stop_batch[bid], max_traj_len, fill_val=-1.0)
#         weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)
    

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
        observations_batch[sensor] = observations_batch[sensor].transpose(1,0)
        observations_batch[sensor] = observations_batch[sensor].contiguous().view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
#     weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(corrected_actions_batch, dtype=torch.float)
    not_done_masks[0] = 0
    oracle_stop_batch = torch.stack(oracle_stop_batch, dim=1)

    prev_actions_batch = prev_actions_batch.transpose(1,0)
    not_done_masks = not_done_masks.transpose(1,0)
    corrected_actions_batch = corrected_actions_batch.transpose(1,0)
#     weights_batch = weights_batch.transpose(1,0)
    oracle_stop_batch = oracle_stop_batch.transpose(1,0)

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.contiguous().view(-1, 2),
        not_done_masks.contiguous().view(-1, 2),
        corrected_actions_batch.contiguous().view(-1,2),
        oracle_stop_batch.contiguous().view(-1,1)
    )

    # return (
    #     observations_batch,
    #     prev_actions_batch,
    #     not_done_masks,
    #     corrected_actions_batch,
    # )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
        is_bert=False
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size
        self.is_bert = is_bert

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.load_ordering.pop()).encode()), raw=False
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions, stop_step = self._load_next()
        # obs, prev_actions, oracle_actions = self._load_next()

        discrete_oracle_actions = obs['vln_oracle_action_sensor'].copy()
        val = int(stop_step[-1])-1
        discrete_oracle_actions[val:]=4
        obs['vln_oracle_action_sensor'] = discrete_oracle_actions
        oracle_stop = np.zeros_like(obs['vln_oracle_action_sensor'])
        oracle_stop[val:] = 1

        if self.is_bert:            
            instruction_batch = obs['instruction'][0]
            instruction_batch = np.expand_dims(instruction_batch, axis=0)
            obs['instruction'] = instruction_batch
            # del obs['glove_tokens']
        else:
            instruction_batch = obs['glove_tokens'][0]
            instruction_batch = np.expand_dims(instruction_batch, axis=0)
            obs['instruction'] = instruction_batch
            del obs['glove_tokens']
        # instruction_batch = obs['instruction'][0]
        # instruction_batch = np.expand_dims(instruction_batch, axis=0)
        # obs['instruction'] = instruction_batch
        for k, v in obs.items():
            obs[k] = torch.from_numpy(v)

        prev_actions = torch.from_numpy(prev_actions)
        oracle_stop = torch.from_numpy(oracle_stop)
        oracle_actions = torch.from_numpy(oracle_actions)
        return (obs, prev_actions, oracle_actions, oracle_stop)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(_block_shuffle(list(range(start, end)), self.preload_size))
        )

        return self


@baseline_registry.register_trainer(name="robo_vln_trainer")
class RoboDaggerTrainer(BaseRLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.envs = None

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.lmdb_features_dir = self.config.DAGGER.LMDB_FEATURES_DIR.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        self.lmdb_eval_dir = self.config.DAGGER.LMDB_EVAL_DIR

    def _setup_actor_critic_agent(
        self, config: Config, load_from_ckpt: bool, ckpt_path: str
    ) -> None:
        r"""Sets up actor critic and agent.
        Args:
            config: MODEL config
        Returns:
            None
        """
        config.defrost()
        config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        config.freeze()

        if config.CMA.use:
            self.actor_critic = CMANet(
                observation_space=self.envs.observation_space,
                num_actions=2,
                model_config=config,
            )
        elif config.SEM_ATTN_ENCODER.use:
            self.actor_critic = Seq2Seq_Sem_Attn_Policy(
                observation_space=self.envs.observation_space,
                action_space=self.envs.action_space,
                model_config=config,
            )
        elif config.LANG_ATTN.use:
            self.actor_critic = Seq2Seq_Lang_Attn(
                observation_space=self.envs.observation_space,
                action_space=self.envs.action_space,
                model_config=config,
            )
        elif config.SEM_TEXT_ATTN.use:
            self.actor_critic = Seq2Seq_Sem_Text_Attn(
                observation_space=self.envs.observation_space,
                action_space=self.envs.action_space,
                model_config=config,
            )

        elif config.TRANSFORMER.use:
            self.actor_critic = TransformerHybridPolicy(
                observation_space=self.envs.observation_space,
                action_space=self.envs.action_space,
                model_config=config,
                batch_size=self.config.DAGGER.BATCH_SIZE,
                gpus = [0,1,2]
            )
        else:
            self.actor_critic = Seq2SeqNet(
                observation_space=self.envs.observation_space,
                num_actions=2,
                num_sub_tasks=self.envs.action_space.n,
                model_config=config,
                batch_size = self.config.DAGGER.BATCH_SIZE,
            )

        if not self.config.MODEL.TRANSFORMER.split_gpus:
            self.actor_critic.to(self.device)    

        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=self.config.DAGGER.LR
        )

        # self.optimizer = torch.optim.AdamW(self.actor_critic.parameters(), 
        #                             lr=self.config.MODEL.TRANSFORMER.lr, 
        #                             weight_decay=self.config.MODEL.TRANSFORMER.weight_decay)

        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=2e-6, max_lr=1e-4, step_size_up=1000,step_size_down=50000, cycle_momentum=False)
        if load_from_ckpt:
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.actor_critic.load_state_dict(ckpt_dict["state_dict"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")
        logger.info("Finished setting up actor critic model.")

    def save_checkpoint(self, file_name) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.actor_critic.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _update_dataset(self, data_it):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        # if self.envs is None:
        #     self.envs = construct_envs_auto_reset_false(self.config, get_env_class(self.config.ENV_NAME))

        prev_actions = np.zeros((1,2))
        done = False
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True
        collected_eps = 0
        # with tqdm.tqdm(total=self.config.DAGGER.UPDATE_SIZE) as pbar, torch.no_grad():
        with tqdm.tqdm(total=self.config.DAGGER.UPDATE_SIZE) as pbar, lmdb.open(
            self.lmdb_features_dir, map_size=int(self.config.DAGGER.LMDB_MAP_SIZE)
        ) as lmdb_env, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)
            stop_step=0
            # episodes = []
            for episode in range(self.config.DAGGER.UPDATE_SIZE):
                episode = []
                observations = self.envs.reset()
                observations = transform_obs(
                    observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, is_bert=self.config.MODEL.INSTRUCTION_ENCODER.is_bert
                )

                reference_path = self.envs.habitat_env.current_episode.reference_path + [
                self.envs.habitat_env.current_episode.goals[0].position
                ]
                continuous_path_follower = ContinuousPathFollower(
                self.envs.habitat_env._sim, reference_path, waypoint_threshold=0.4)

                is_done = False
                steps=0
                stop_flag = False
                valid_trajectories = True
                while continuous_path_follower.progress < 1.0:
                    steps+=1
                    if is_done:
                        break
                    continuous_path_follower.update_waypoint()
                    agent_state = self.envs.habitat_env._sim.get_agent_state()
                    previous_rigid_state = habitat_sim.RigidState(
                    quat_to_magnum(agent_state.rotation), agent_state.position
                    )

                    if np.isnan(continuous_path_follower.waypoint).any() or np.isnan(previous_rigid_state.translation).any() or np.isnan(quaternion.as_euler_angles(quat_from_magnum(previous_rigid_state.rotation))).any():
                        valid_trajectories = False
                        break
                    vel,omega = track_waypoint(
                        continuous_path_follower.waypoint,
                        previous_rigid_state,
                        vel_control,
                        progress = continuous_path_follower.progress,
                        dt=self.config.DAGGER.time_step,
                    )
                    observations, _, done, _ = self.envs.step(vel_control)
                    episode_over, success = done

                    if continuous_path_follower.progress >0.985 and not stop_flag:
                        stop_step = steps
                        stop_flag = True

                    is_done = episode_over or (success and abs(vel)<0.005)

                    observations = transform_obs(
                        observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, is_bert=self.config.MODEL.INSTRUCTION_ENCODER.is_bert
                    )
                    actions = np.expand_dims(np.asarray([vel,omega]), axis=0)
                    episode.append(
                        (
                            observations,
                            prev_actions,
                            actions,
                            stop_step

                        )
                    )
                    prev_actions = actions

                # episodes.append(episode)

                # Save episode to LMDB directory
                if valid_trajectories:
                    traj_obs = batch_obs_data_collect([step[0] for step in episode],  device=torch.device("cpu"))
                    for k, v in traj_obs.items():
                        traj_obs[k] = v.numpy()
                    transposed_ep = [
                        traj_obs,
                        np.array([step[1] for step in episode], dtype=float),
                        np.array([step[2] for step in episode], dtype=float),
                        [step[3] for step in episode],
                    ]
                    txn.put(
                        str(start_id + collected_eps).encode(),
                        msgpack_numpy.packb(transposed_ep, use_bin_type=True),
                    )

                    pbar.update()
                    collected_eps += 1

                if (
                    collected_eps % self.config.DAGGER.LMDB_COMMIT_FREQUENCY
                ) == 0:
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

                episode = []
                prev_actions = np.zeros((1,2))
            txn.commit()
        self.envs.close()
        self.envs = None

    def _update_agent(
        self, observations, prev_actions, not_done_masks, corrected_actions, oracle_stop,recurrent_hidden_states
    ):
        # T, N, C = corrected_actions.size()
        self.optimizer.zero_grad()
        criterion = nn.MSELoss()
        sub_goal_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        stop_criterion = nn.BCEWithLogitsLoss()

        AuxLosses.clear()

        recurrent_hidden_states = repackage_hidden(recurrent_hidden_states)

        batch = (observations, recurrent_hidden_states, prev_actions, not_done_masks)
        del recurrent_hidden_states, prev_actions, not_done_masks
        
        if self.config.MODEL.FLAT_AUX_LOSS.use:
            output, stop_out, sub_goal_out, recurrent_hidden_states = self.actor_critic(batch)

            high_level_action_mask = observations['vln_oracle_action_sensor'] ==0
            sub_goal_out = sub_goal_out.masked_fill_(high_level_action_mask, 0)
            observations['vln_oracle_action_sensor'] = observations['vln_oracle_action_sensor'].squeeze(1).to(dtype=torch.int64)
            sub_goal_loss = sub_goal_criterion(sub_goal_out,(observations['vln_oracle_action_sensor']-1))
        else:
            output, stop_out, recurrent_hidden_states = self.actor_critic(batch)

        del observations
        action_mask = corrected_actions==0
        output = output.masked_fill_(action_mask, 0)
        output = output.to(dtype=torch.float)
        corrected_actions = corrected_actions.to(dtype=torch.float)
        action_loss = criterion(output, corrected_actions)
        

        mask = (oracle_stop!=-1)
        oracle_stop = torch.masked_select(oracle_stop, mask)
        stop_out = torch.masked_select(stop_out, mask)
        stop_loss   = stop_criterion(stop_out, oracle_stop)


        if self.config.MODEL.FLAT_AUX_LOSS.use:
            loss = action_loss + stop_loss + sub_goal_loss
        else: 
            aux_mask = ~action_mask[:,0]
            aux_loss = AuxLosses.reduce(aux_mask)
            loss = action_loss + stop_loss +aux_loss

        loss.backward()
        self.optimizer.step()
        loss = (action_loss.detach().item(), stop_loss.detach().item(), aux_loss.item())
        del output, action_loss
        return loss, recurrent_hidden_states

    def _update_agent_val(
        self, observations, prev_actions, not_done_masks, corrected_actions, oracle_stop,recurrent_hidden_states
    ):
        criterion = nn.MSELoss()
        stop_criterion = nn.BCEWithLogitsLoss()
        sub_goal_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        AuxLosses.clear()
        recurrent_hidden_states = repackage_hidden(recurrent_hidden_states)

        batch = (observations, recurrent_hidden_states, prev_actions, not_done_masks)
        del recurrent_hidden_states, prev_actions, not_done_masks
        


        if self.config.MODEL.FLAT_AUX_LOSS.use:
            output, stop_out, sub_goal_out, recurrent_hidden_states = self.actor_critic(batch)
            high_level_action_mask = observations['vln_oracle_action_sensor'] ==0
            sub_goal_out = sub_goal_out.masked_fill_(high_level_action_mask, 0)
            observations['vln_oracle_action_sensor'] = observations['vln_oracle_action_sensor'].squeeze(1).to(dtype=torch.int64)
            sub_goal_loss = sub_goal_criterion(sub_goal_out,(observations['vln_oracle_action_sensor']-1))
            predicted = torch.argmax(sub_goal_out, dim=1)
            corrected_mask = ~high_level_action_mask
            correct = torch.masked_select((observations['vln_oracle_action_sensor']-1), corrected_mask)
            predicted = torch.masked_select(predicted, corrected_mask)
            accuracy = (predicted == correct).sum().item()
            total = predicted.size(0)
        else: 
            output, stop_out, recurrent_hidden_states = self.actor_critic(batch)

        action_mask = corrected_actions==0
        output = output.masked_fill_(action_mask, 0)
        output = output.to(dtype=torch.float)
        corrected_actions = corrected_actions.to(dtype=torch.float)
        action_loss = criterion(output, corrected_actions)

        mask = (oracle_stop!=-1)
        oracle_stop = torch.masked_select(oracle_stop, mask)
        stop_out = torch.masked_select(stop_out, mask)
        stop_loss   = stop_criterion(stop_out, oracle_stop)
        # loss = action_loss + stop_loss

        aux_mask = ~action_mask[:,0]
        aux_loss = AuxLosses.reduce(aux_mask)
        loss = (action_loss.item(), stop_loss.item(), aux_loss.item())

        if self.config.MODEL.FLAT_AUX_LOSS.use:
            return loss, recurrent_hidden_states, accuracy, total
        else:
            return loss, recurrent_hidden_states


    def train_epoch(self, diter, length, batch_size, epoch, writer, train_steps):
        loss, action_loss, aux_loss = 0, 0, 0
        # action_losses=[]
        # stop_losses =[]
        # total_losses=[]
        self.actor_critic.train()
        for batch in tqdm.tqdm(
            diter, total=length // batch_size, leave=False
        ):

            (   observations_batch,
                prev_actions_batch,
                not_done_masks,
                corrected_actions_batch,
                oracle_stop_batch
            ) = batch

            # recurrent_hidden_states = torch.zeros(
            #     self.actor_critic.state_encoder.num_recurrent_layers,
            #     self.config.DAGGER.BATCH_SIZE,
            #     self.config.MODEL.STATE_ENCODER.hidden_size,
            #     device=self.device,
            # )

            recurrent_hidden_states = torch.zeros(
                self.actor_critic.num_recurrent_layers,
                self.config.DAGGER.BATCH_SIZE,
                self.config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            )
            batch_split = split_batch_tbptt(observations_batch, prev_actions_batch, not_done_masks, 
                    corrected_actions_batch, oracle_stop_batch, self.config.DAGGER.tbptt_steps, 
                    self.config.DAGGER.split_dim)

            # batch_split = split_batch_tbptt(observations_batch, prev_actions_batch, not_done_masks, 
            #                                 corrected_actions_batch, self.config.DAGGER.tbptt_steps, 
            #                                 self.config.DAGGER.split_dim)

            del observations_batch, prev_actions_batch, not_done_masks, corrected_actions_batch, batch
            for split in batch_split:
                (   observations_batch,
                    prev_actions_batch,
                    not_done_masks,
                    corrected_actions_batch,
                    oracle_stop_batch
                ) = split                        
            # for split in batch_split:
            #     (   observations_batch,
            #         prev_actions_batch,
            #         not_done_masks,
            #         corrected_actions_batch,
            #     ) = repackage_mini_batch(split)

                observations_batch = {
                    k: v.to(device=self.device, non_blocking=True)
                    for k, v in observations_batch.items()
                }
                try:
                    output, recurrent_hidden_states= self._update_agent(
                        observations_batch,
                        prev_actions_batch.to(
                            device=self.device, non_blocking=True
                        ),
                        not_done_masks.to(
                            device=self.device, non_blocking=True
                        ),
                        corrected_actions_batch.to(
                            device=self.device, non_blocking=True
                        ),
                        oracle_stop_batch.to(
                                device=self.device, non_blocking=True
                        ),
                        recurrent_hidden_states,
                    )
                    writer.add_scalar(f"Action Loss", output[0], train_steps)
                    writer.add_scalar(f"Stop Loss", output[1], train_steps)

                    if self.config.MODEL.FLAT_AUX_LOSS.use:
                        writer.add_scalar(f"Sub Goal Loss", output[2], train_steps)
                        writer.add_scalar(f"Total Loss", output[0] + output[1] + output[2], train_steps)
                    else: 
                        writer.add_scalar(f"Aux Loss", output[2], train_steps)
                        writer.add_scalar(f"Total Loss", output[0] + output[1]+output[2], train_steps)
                    train_steps += 1
                    # action_losses.append(output[0])
                    # stop_losses.append(output[1])
                    # total_losses.append(output[0] + output[1])
                    # aux_loss += output[2]
                except:
                    logger.info(
                        "ERROR: failed to update agent. Updating agent with batch size of 1."
                    )
                    loss, action_loss, aux_loss = 0, 0, 0
                    prev_actions_batch = prev_actions_batch.cpu()
                    not_done_masks = not_done_masks.cpu()
                    corrected_actions_batch = corrected_actions_batch.cpu()
                    weights_batch = weights_batch.cpu()
                    observations_batch = {
                        k: v.cpu() for k, v in observations_batch.items()
                    }

                    for i in range(not_done_masks.size(0)):
                        output = self._update_agent(
                            {
                                k: v[i].to(
                                    device=self.device, non_blocking=True
                                )
                                for k, v in observations_batch.items()
                            },
                            prev_actions_batch[i].to(
                                device=self.device, non_blocking=True
                            ),
                            not_done_masks[i].to(
                                device=self.device, non_blocking=True
                            ),
                            corrected_actions_batch[i].to(
                                device=self.device, non_blocking=True
                            ),
                            weights_batch[i].to(
                                device=self.device, non_blocking=True
                            ),
                        )
                        loss += output[0]
                        action_loss += output[1]
                        aux_loss += output[2]

                # # For CyclicalLR
                # self.scheduler.step()

            # logger.info(f"train_action_loss: {np.mean(action_losses)}")
            # logger.info(f"train_stop_loss: {np.mean(stop_losses)}")
            # logger.info(f"Train_combined_loss: {np.mean(total_losses)}")
            # # logger.info(f"train_action_loss: {action_loss}")
            # logger.info(f"train_aux_loss: {aux_loss}")
            # logger.info(f"Batches processed: {train_steps}.")
            # logger.info(f"Epoch {epoch}.")
            # wandb.log({
            # "Train Action Loss": np.mean(action_losses),
            # "Train Stop Loss": np.mean(stop_losses),
            # "Train Total Loss": np.mean(total_losses),
            # "Test Aux Loss": aux_loss}, step = train_steps)

            # writer.add_scalar(f"train_loss_iter_{dagger_it}", losses.mean(), step_id)
            # writer.add_scalar(
            #     f"train_aux_loss_iter_{dagger_it}", aux_loss, step_id
            # )
            # writer.add_scalar(
            #     f"Cache_0_mem_usage_{dagger_it}", torch.cuda.memory_cached(0), step_id
            # )

        self.save_checkpoint(
            f"ckpt.{self.config.DAGGER.EPOCHS + epoch}.pth"
        )
        return train_steps


    def val_epoch(self, diter, length, batch_size, epoch, writer, val_steps):
        loss, action_loss, aux_loss = 0, 0, 0
        val_losses =[]
        self.actor_critic.eval()

        correct_labels = 0
        total_correct=0

        with torch.no_grad():
            for batch in tqdm.tqdm(
                diter, total=length // batch_size, leave=False
            ):

                (   observations_batch,
                    prev_actions_batch,
                    not_done_masks,
                    corrected_actions_batch,
                    oracle_stop_batch
                ) = batch

                recurrent_hidden_states = torch.zeros(
                    self.actor_critic.num_recurrent_layers,
                    self.config.DAGGER.BATCH_SIZE,
                    self.config.MODEL.STATE_ENCODER.hidden_size,
                    device=self.device,
                )
                batch_split = split_batch_tbptt(observations_batch, prev_actions_batch, not_done_masks, 
                        corrected_actions_batch, oracle_stop_batch, self.config.DAGGER.tbptt_steps, 
                        self.config.DAGGER.split_dim)

                del observations_batch, prev_actions_batch, not_done_masks, corrected_actions_batch, batch
                for split in batch_split:
                    (   observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        corrected_actions_batch,
                        oracle_stop_batch
                    ) = split                        
                    observations_batch = {
                        k: v.to(device=self.device, non_blocking=True)
                        for k, v in observations_batch.items()
                    }
                    output, recurrent_hidden_states= self._update_agent_val(
                        observations_batch,
                        prev_actions_batch.to(
                            device=self.device, non_blocking=True
                        ),
                        not_done_masks.to(
                            device=self.device, non_blocking=True
                        ),
                        corrected_actions_batch.to(
                            device=self.device, non_blocking=True
                        ),
                        oracle_stop_batch.to(
                                device=self.device, non_blocking=True
                        ),
                        recurrent_hidden_states,
                    )

                    # correct_labels+= accuracy 
                    # total_correct+=total

                    writer.add_scalar(f"Val Action Loss", output[0], val_steps)
                    writer.add_scalar(f"Val Stop Loss", output[1], val_steps)
                    writer.add_scalar(f"Aux Loss", output[2], val_steps)
                    writer.add_scalar(f"Val Total Loss", output[0]+output[1]+ output[2], val_steps)
                    val_steps += 1
                    val_losses.append(output[0] + output[1] + output[2])

                # logger.info(f"val_loss: {np.mean(losses_time)}")
                # logger.info(f"Val_aux_loss: {aux_loss}")
                # logger.info(f"Batches processed: {val_steps}.")
                # logger.info(f"Epoch {epoch}.")
                # wandb.log({
                # "Val Loss iter": np.mean(losses_time),
                # "Val Aux Loss iter": aux_loss}, step=val_steps)
                
                # writer.add_scalar(f"Val_losses_time_iter_{dagger_it}", losses_time.mean(), step_id)
                # writer.add_scalar(
                #     f"val_aux_loss_iter_{dagger_it}", aux_loss, step_id
                # )
                # end = time.time()
            # writer.add_scalar(f"Val_losses_iter_{dagger_it}", val_losses.mean(), epoch)

            # final_accuracy = 100 * correct_labels / total_correct
            writer.add_scalar(f"Val Loss Epoch", np.mean(val_losses), val_steps)
            # writer.add_scalar(f"Validation Accuracy", final_accuracy, epoch)
            return val_steps

            

    def train(self) -> None:
        r"""Main method for training DAgger.

        Returns:
            None
        """
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)

        if self.config.DAGGER.PRELOAD_LMDB_FEATURES:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
                lmdb.open(self.lmdb_eval_dir, readonly=True)
            except lmdb.Error as err:
                logger.error("Cannot open database for teacher forcing preload.")
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir, map_size=int(self.config.DAGGER.LMDB_MAP_SIZE)
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.DAGGER.P == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        # config.TASK_CONFIG.DATASET.SPLIT = config.DAGGER.COLLECT_DATA_SPLIT
        self.config.freeze()

        if self.config.DAGGER.PRELOAD_LMDB_FEATURES:
            # when preloadeding features, its quicker to just load one env as we just
            # need the observation space from it.
            single_proc_config = self.config.clone()
            single_proc_config.defrost()
            single_proc_config.NUM_PROCESSES = 1
            single_proc_config.freeze()
            self.envs = construct_env(self.config)
        else:
            self.envs = construct_env(self.config)

        self._setup_actor_critic_agent(
            self.config.MODEL,
            self.config.DAGGER.LOAD_FROM_CKPT,
            self.config.DAGGER.CKPT_TO_LOAD,
        )

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.actor_critic.parameters())
            )
        )
        logger.info(
            "agent number of trainable parameters: {}".format(
                sum(
                    p.numel() for p in self.actor_critic.parameters() if p.requires_grad
                )
            )
        )

        if self.config.DAGGER.PRELOAD_LMDB_FEATURES:
            self.envs.close()
            del self.envs
            self.envs = None

        with TensorboardWriter(
            wandb.run.dir, flush_secs=self.flush_secs, purge_step=0
        ) as writer:
            for dagger_it in range(self.config.DAGGER.ITERATIONS):
                if not self.config.DAGGER.PRELOAD_LMDB_FEATURES:
                    self._update_dataset(
                        dagger_it + (1 if self.config.DAGGER.LOAD_FROM_CKPT else 0)
                    )

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                dataset = IWTrajectoryDataset(
                    self.lmdb_features_dir,
                    self.config.DAGGER.USE_IW,
                    inflection_weight_coef=self.config.MODEL.inflection_weight_coef,
                    lmdb_map_size=self.config.DAGGER.LMDB_MAP_SIZE,
                    batch_size=self.config.DAGGER.BATCH_SIZE,
                    is_bert = self.config.MODEL.INSTRUCTION_ENCODER.is_bert,
                )
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.DAGGER.BATCH_SIZE,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=1,
                )

                dataset_eval = IWTrajectoryDataset(
                    self.lmdb_eval_dir,
                    self.config.DAGGER.USE_IW,
                    inflection_weight_coef=self.config.MODEL.inflection_weight_coef,
                    lmdb_map_size=self.config.DAGGER.LMDB_EVAL_SIZE,
                    batch_size=self.config.DAGGER.BATCH_SIZE,
                    is_bert = self.config.MODEL.INSTRUCTION_ENCODER.is_bert,
                )
                diter_eval = torch.utils.data.DataLoader(
                    dataset_eval,
                    batch_size=self.config.DAGGER.BATCH_SIZE,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=1,
                )
                train_steps =0
                val_steps =0

                AuxLosses.activate()
                print("starting training loop")
                for epoch in tqdm.trange(self.config.DAGGER.EPOCHS):
                    train_steps = self.train_epoch(diter, dataset.length, dataset.batch_size, epoch, writer, train_steps)
                    val_steps = self.val_epoch(diter_eval, dataset_eval.length, dataset_eval.batch_size, epoch, writer, val_steps)
                AuxLosses.deactivate()

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))

            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            if recurrent_hidden_states:
                recurrent_hidden_states = recurrent_hidden_states[:, state_index]
            # recurrent_hidden_states = recurrent_hidden_states
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

        return (envs, recurrent_hidden_states, not_done_masks, prev_actions, batch)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def draw_top_down_map(self, info, output_size):
        return maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], output_size
        )

    def _eval_checkpoint(
        self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0
    ) -> None:
        r"""Evaluates a single checkpoint. Assumes episode IDs are unique.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")["config"]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.NDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.SDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")

        config.freeze()

        gt_path = config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=config.TASK_CONFIG.DATASET.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_json = json.load(f)

        # setup agent
        self.envs =  construct_env(config)
        self.device = (
            torch.device("cuda", config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._setup_actor_critic_agent(config.MODEL, True, checkpoint_path)

        vc = habitat_sim.physics.VelocityControl()
        vc.controlling_lin_vel = True
        vc.lin_vel_is_local = True
        vc.controlling_ang_vel = True
        vc.ang_vel_is_local = True

        observations = self.envs.reset()
        observations = transform_obs(
            observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, is_bert=self.config.MODEL.INSTRUCTION_ENCODER.is_bert
        )

        # observations = {
        #     k: torch.tensor(v).to(device=self.device, non_blocking=True)
        #     for k, v in observations.items()
        # }
        observations = batch_obs(observations, self.device)
        # batch["instruction_batch"] = batch['instruction']
        # del batch['instruction']

        eval_recurrent_hidden_states = torch.zeros(
            self.actor_critic.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            config.NUM_PROCESSES, 2, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(config.NUM_PROCESSES, 2, device=self.device)

        stats_episodes = {}  # dict of dicts that stores stats per episode

        if len(config.VIDEO_OPTION) > 0:
            rgb_frames = []
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
            # rgb_frames = [[] for _ in range(config.NUM_PROCESSES)]

        if config.PLOT_ATTENTION:
            attention_weights = [[] for _ in range(config.NUM_PROCESSES)]
            save_actions = [[] for _ in range(config.NUM_PROCESSES)]

        self.actor_critic.eval()
        k=0
        ep_count = 0
        min_2nd_dim = 1000
        steps=0
        locations=[]
        action_dict = {0:'stop', 1: 'straight', 2: 'left', 3:'right'}

        while (
            len(stats_episodes) < config.EVAL.EPISODE_COUNT
        ):
            current_episode = self.envs.habitat_env.current_episode
            # print("Number of episodes:", self.envs.number_of_episodes)
            # print("Number of envs:", self.envs.num_envs)
            # # print("Count episodes:", self.envs.count_episodes)
            # print("--------------------------------------------")
            is_done = False
            locations.append(self.envs.habitat_env._sim.get_agent_state().position.tolist())
            with torch.no_grad():
                batch = (observations, eval_recurrent_hidden_states, prev_actions, not_done_masks)
                output, stop_out, _, eval_recurrent_hidden_states = self.actor_critic(batch)
                # (_, actions, _, eval_recurrent_hidden_states) = self.actor_critic.act(
                #     batch,
                #     eval_recurrent_hidden_states,
                #     prev_actions,
                #     not_done_masks,
                #     deterministic=True,
                # )
                prev_actions = output

            lin_vel = output[:, 0]

            not_done_masks = torch.ones(config.NUM_PROCESSES, 2, device=self.device)

            # print("Loop:", k, "hidden states shape:", eval_recurrent_hidden_states.shape)

            # print("output",output)
            # print("output.shape",output.shape)
            # print("v", output[:,0])

            
            vc.linear_velocity = mn.Vector3(0, 0, output[:,0].cpu().numpy())
            max_turn_speed = 1.0
            vc.angular_velocity = mn.Vector3(0, np.clip(output[:,1].cpu().numpy(), -max_turn_speed, max_turn_speed), 0)
            observations, _, done, info = self.envs.step(vc) 
            episode_over, success = done

            stop_pred = torch.round(torch.sigmoid(stop_out))
            episode_success = success and (lin_vel<0.25 or stop_pred ==1)
            is_done = episode_over or episode_success 
            steps+=1

            dirname = config.VIDEO_DIR

            best_action = np.asscalar(observations['vln_oracle_action_sensor'])

            if not os.path.exists(dirname+"/"+str(current_episode.episode_id)):
                os.makedirs(dirname+"/"+str(current_episode.episode_id))
            map1 = self.draw_top_down_map(info, 1024)
            # map1 = Image.fromarray(map1)
            map_dir = dirname+"/"+str(current_episode.episode_id)+"/"+"map"+str(steps)+'.jpg'
            plt.imsave(map_dir,map1)
            im = observations['rgb']
            # added_image = im
            # im= np.asarray(im)
            overlay = cv2.imread('/home/mirshad7/robo-vln/'+action_dict[best_action]+'.png')
            overlay = cv2.resize(overlay, (224, 224))
            added_image = cv2.addWeighted(im,1,overlay,1,0)
            # added_image = Image.fromarray(added_image)
            # im = Image.fromarray(im, mode="RGBA")
            im_dir = dirname+"/"+str(current_episode.episode_id)+"/"+"im"+str(steps)+'.jpg'
            plt.imsave(im_dir,added_image)

            depth_obs = observations['depth'].squeeze(2)
            depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
            depth_dir = dirname+"/"+str(current_episode.episode_id)+"/"+"depth"+str(steps)+'.jpg'
            plt.imsave(depth_dir,depth_img)

            if len(config.VIDEO_OPTION) > 0:
                frame = observations_to_image(observations, info)
                frame = append_text_to_image(
                    frame, current_episode.instruction.instruction_text
                )
                rgb_frames.append(frame)
            # print("dones",dones, "\n")
            # print("not_done_masks", not_done_masks)

            # not_done_masks = torch.tensor(
            #     [[0.0] if done else [1.0] for done in dones],
            #     dtype=torch.float,
            #     device=self.device,
            # )
            # reset envs and observations if necessary

            # for i in range(self.envs.num_envs):
                # print("loop number", k)
                # print("current episode",self.envs.current_episodes()[i].episode_id)
                # print("------------------------------------------------")
            # for i in range(self.envs.num_envs):
            #     if config.PLOT_ATTENTION:
            #         attention_weights[i].append(batch['lang_attention'][i])
            #         min_dim = batch['lang_attention'][i].shape[1]
            #         min_2nd_dim = np.minimum(min_2nd_dim,min_dim)
            #         save_actions[i].append(actions[i])
            #     if len(config.VIDEO_OPTION) > 0:

            #         frame = observations_to_image(observations[i], infos[i])
            #         frame = append_text_to_image(
            #             frame, current_episodes[i].instruction.instruction_text
            #         )
            #         rgb_frames[i].append(frame)

            if is_done or steps==self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS:
                # calulcate NDTW here 
                gt_locations = self.gt_json[str(current_episode.episode_id)]["locations"]
                dtw_distance = fastdtw(locations, gt_locations, dist=self._euclidean_distance)[0]
                nDTW = np.exp(-dtw_distance / (len(gt_locations) * config.TASK_CONFIG.TASK.NDTW.SUCCESS_DISTANCE))
                # print("dtw time",time.time()-dtw_time)
                locations=[]
                is_done = False
                # print("Episode complete in steps:", steps)
                # print("One Step Time", time.time()-start_time)
                ep_count+=1
                steps=0
                print("dones:", done)
                stats_episodes[current_episode.episode_id] = info
                stats_episodes[current_episode.episode_id]['ndtw'] = nDTW
                
                print("Current episode ID:", current_episode)
                print("Episode Completed:", ep_count)
                print(" Episode done---------------------------------------------")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("stats_episode", len(stats_episodes))
                observations = self.envs.reset()
                prev_actions = torch.zeros(
                    config.NUM_PROCESSES, 2, device=self.device, dtype=torch.long
                )
                not_done_masks = torch.zeros(config.NUM_PROCESSES, 2, device=self.device)

                eval_recurrent_hidden_states = torch.zeros(
                    self.actor_critic.num_recurrent_layers,
                    self.config.NUM_PROCESSES,
                    self.config.MODEL.STATE_ENCODER.hidden_size,
                    device=self.device,
                )
                metrics={"SPL":round(
                            stats_episodes[current_episode.episode_id]["spl"], 6
                        ) } 
                if len(config.VIDEO_OPTION) > 0:
                    time_step=30
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames,
                        episode_id=current_episode.episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics=metrics,
                        tb_writer=writer,
                        fps = int (1.0/time_step),
                    )
                    del stats_episodes[current_episode.episode_id]["top_down_map"]
                    rgb_frames =[]
                # del stats_episodes[current_episode.episode_id]["top_down_map"]
                if config.PLOT_ATTENTION:
                    for j in range(len(attention_weights[i])):
                        attention_weights[i][j] = attention_weights[i][j][:,:min_2nd_dim]
                    attention_weights[i]= torch.cat(attention_weights[i], dim=0).cpu().numpy()
                    attention_to_image(
                            image_dir = config.VIDEO_DIR,
                            attention = attention_weights[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=metrics,
                            actions = save_actions[i]
                        )
                    attention_weights[i] = [] 
                    save_actions[i] =[]
            observations = transform_obs(
                observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, is_bert=self.config.MODEL.INSTRUCTION_ENCODER.is_bert
            )
            observations = batch_obs(observations, self.device)
            k+=1
            
            # for i in range(self.envs.num_envs):
            #     if next_episodes[i].episode_id in stats_episodes:
            #         envs_to_pause.append(i)

            # (
            #     self.envs,
            #     eval_recurrent_hidden_states,
            #     not_done_masks,
            #     prev_actions,
            #     batch,
            # ) = self._pause_envs(
            #     envs_to_pause,
            #     self.envs,
            #     eval_recurrent_hidden_states,
            #     not_done_masks,
            #     prev_actions,
            #     batch,
            # )

        self.envs.close()

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

        split = config.TASK_CONFIG.DATASET.SPLIT
        os.makedirs(config.EVAL.VAL_LOG_DIR, exist_ok=True)
        val_log_path = os.path.join(config.EVAL.VAL_LOG_DIR,f"stats_ckpt_{checkpoint_index}_{split}.json")
        with open(val_log_path, "w") as f:
            json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)
