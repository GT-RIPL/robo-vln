import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from vlnce_baselines.models.seq2seq_highlevel_cross_modal import Seq2Seq_HighLevel
from vlnce_baselines.models.seq2seq_lowlevel_ic import Seq2Seq_LowLevel
# from vlnce_baselines.models.policy import BasePolicy
from collections import defaultdict
import time

class HierarchicalCMANet(nn.Module):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.

    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, num_actions: int, num_sub_tasks:int, model_config: Config, batch_size: int):
        super().__init__()

        self.device1 = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.device2 = (
            torch.device("cuda:1")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.batch_size = batch_size
        self.model_config = model_config

        self.high_level = Seq2Seq_HighLevel(
                observation_space=observation_space,
                num_actions=num_actions,
                model_config=model_config,
                batch_size = batch_size,
            )

        self.low_level = Seq2Seq_LowLevel(
                observation_space=observation_space,
                num_actions=2,
                num_sub_tasks=num_sub_tasks,
                model_config=model_config,
                batch_size = batch_size,
            )

        self.high_level.to(self.device1)
        self.low_level.to(self.device2)


    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        pass

    @property
    def num_recurrent_layers(self):
        return self.high_level.state_encoder.num_recurrent_layers

    # def get_single_obs(self, observations, prev_actions, masks, discrete_actions, index):
    #     obs = defaultdict()
    #     for sensor in observations:
    #         if sensor =='instruction':
    #             obs[sensor] = observations[sensor]
    #             continue
    #         obs[sensor] = observations[sensor][index].unsqueeze(0)
    #     prev_actions_single = prev_actions[index].unsqueeze(0)
    #     masks_single = masks[index].unsqueeze(0)
    #     discrete_actions_single = discrete_actions[index].unsqueeze(0)

    #     return obs, prev_actions_single, masks_single, discrete_actions_single

    def get_single_obs(self, high_state, low_state, prev_actions, masks, index):
        # obs = defaultdict()
        # for sensor in observations:
        #     if sensor =='instruction':
        #         obs[sensor] = observations[sensor]
        #         continue
        #     obs[sensor] = observations[sensor][index].unsqueeze(0)

        hs = high_state[index].unsqueeze(0)
        ls = low_state[index].unsqueeze(0)
        # print("obs",obs.shape)
        prev_actions_single = prev_actions[index].unsqueeze(0)
        masks_single = masks[index].unsqueeze(0)
        # discrete_actions_single = discrete_actions[index].unsqueeze(0)

        return hs, ls, prev_actions_single, masks_single


    def forward(self, batch):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        observations, high_recurrent_hidden_states, low_recurrent_hidden_states, prev_actions, masks, discrete_actions, detached_state_low = batch
        max_len = int(prev_actions.size(0)/self.batch_size)
        high_output=[]
        low_output=[]
        low_stop_output=[]
        rgb_d, lang_encoding = self.high_level.forward_vnl(observations)
        observations = {
            k: v.to(device=self.device2, non_blocking=True)
            for k, v in observations.items()
        }
        low_state = self.low_level.forward_vnl(observations, discrete_actions.to(self.device2, non_blocking=True))
        progress = observations['progress']
        del observations
        for i in range(max_len):
            # if self.batch_size!=max_len: # only works for Batch Size=1
            high_state_single, low_state_single, prev_actions_single, masks_single = self.get_single_obs(rgb_d, low_state, prev_actions, masks, i)
            high_level_output, high_recurrent_hidden_states, detached_state_high = self.high_level(high_state_single, 
                                                                                                   lang_encoding,
                                                                                                   progress, 
                                                                                                   high_recurrent_hidden_states, 
                                                                                                   prev_actions_single, 
                                                                                                   masks_single, 
                                                                                                   detached_state_low)
            # obs_single = {
            # k: v.to(device=self.device2, non_blocking=True)
            # for k, v in obs_single.items()
            # }
            low_level_output, stop_out, low_recurrent_hidden_states, detached_state_low =  self.low_level(low_state_single,
                                                                            progress.to
                                                                            (self.device2,non_blocking=True),
                                                                            low_recurrent_hidden_states,
                                                                            prev_actions.to(
                                                                                device=self.device2, non_blocking=True
                                                                            ),
                                                                            masks_single.to(
                                                                                device=self.device2, non_blocking=True
                                                                            ),
                                                                            detached_state_high.unsqueeze(0).to(device=self.device2, non_blocking=True))
            detached_state_low = detached_state_low.unsqueeze(0).to(self.device1,non_blocking=True)
            high_output.append(high_level_output)
            low_output.append(low_level_output)
            low_stop_output.append(stop_out)
        high_output = torch.cat(high_output,dim=0)
        low_output = torch.cat(low_output,dim=0)  
        low_stop_output = torch.cat(low_stop_output,dim=0)

        high_output= high_output.view(max_len, -1)
        low_output= low_output.view(max_len, -1)
        low_stop_output= low_stop_output.view(max_len, -1)
        return high_output, low_output, low_stop_output, high_recurrent_hidden_states, low_recurrent_hidden_states, detached_state_low
