import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
# from habitat_baselines.rl.ppo.policy import Net

from robo_vln_baselines.common.aux_losses import AuxLosses
from robo_vln_baselines.models.encoders.language_encoder import LanguageEncoder
from robo_vln_baselines.models.encoders.instruction_encoder import InstructionEncoder
from robo_vln_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from robo_vln_baselines.models.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
# from robo_vln_baselines.models.policy import BasePolicy

class Seq2Seq_LowLevel(nn.Module):
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
        self.model_config = model_config
        self.batch_size = batch_size
        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )   
        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
            self.depth_encoder = SimpleDepthCNN(
                observation_space, model_config.DEPTH_ENCODER.output_size
            )
        elif model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "SimpleRGBCNN",
            "TorchVisionResNet50",
        ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

        if model_config.RGB_ENCODER.cnn_type == "SimpleRGBCNN":
            self.rgb_encoder = SimpleRGBCNN(
                observation_space, model_config.RGB_ENCODER.output_size
            )
        elif model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, 
                model_config.RGB_ENCODER.output_size, 
                model_config.RGB_ENCODER.resnet_output_size, 
                device
            )

        self.sub_task_embedding = nn.Embedding(num_sub_tasks+1, 32, padding_idx=4)

        # Init the RNN state decoder
        rnn_input_size = (
            + model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
            + self.sub_task_embedding.embedding_dim
        )

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.progress_monitor = nn.Linear(
            self.model_config.STATE_ENCODER.hidden_size, 1
        )

        self._init_layers()
        self.linear = nn.Linear(self.model_config.STATE_ENCODER.hidden_size, num_actions)
        self.stop_linear = nn.Linear(self.model_config.STATE_ENCODER.hidden_size, 1)

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _init_layers(self):
        nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
        nn.init.constant_(self.progress_monitor.bias, 0)

    def forward(self, batch):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        observations, rnn_hidden_states, prev_actions, masks, discrete_actions = batch
        del batch

        # instructions = self.pad_instructions(observations)
        # del observations['instruction']
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        # if self.model_config.ablate_instruction:
        #     instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        # discrete_action_mask = discrete_actions ==0
        # discrete_actions = (discrete_actions-1).masked_fill_(discrete_action_mask, 4)

        # sub_tasks_embedding = self.sub_task_embedding(discrete_actions.view(-1))
        sub_tasks_embedding = self.sub_task_embedding(discrete_actions)

        x = torch.cat([depth_embedding, rgb_embedding, sub_tasks_embedding], dim=1)
        del depth_embedding, rgb_embedding
        masks = masks[:,0]

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1), observations["progress"], reduction="none"
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )

        out = self.linear(x)
        stop_out = self.stop_linear(x)
        return out, stop_out, rnn_hidden_states
