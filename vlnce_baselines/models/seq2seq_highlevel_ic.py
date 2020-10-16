import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
# from habitat_baselines.rl.ppo.policy import Net
import time

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.language_encoder import LanguageEncoder
from vlnce_baselines.models.encoders.instruction_encoder import InstructionEncoder
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from vlnce_baselines.models.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from vlnce_baselines.models.transformer.transformer import InterModuleAttnDecoder
# from vlnce_baselines.models.policy import BasePolicy

class Seq2Seq_HighLevel(nn.Module):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.

    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, num_actions: int, model_config: Config, batch_size: int):
        super().__init__()
        self.model_config = model_config
        self.batch_size = batch_size
        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Init the instruction encoder

        if model_config.INSTRUCTION_ENCODER.is_bert:
            self.instruction_encoder = LanguageEncoder(model_config.INSTRUCTION_ENCODER, device)
        else:
            self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)    
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

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Init the RNN state decoder
        rnn_input_size = (
            self.instruction_encoder.output_size
            + model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
        )

        if model_config.SEQ2SEQ.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.progress_monitor = nn.Linear(
            self.model_config.STATE_ENCODER.hidden_size, 1
        )
        self.inter_module_attn = InterModuleAttnDecoder(model_config.INTER_MODULE_ATTN)
        self.fc = nn.Linear(self.model_config.STATE_ENCODER.hidden_size*2, self.model_config.STATE_ENCODER.hidden_size)
        self.linear = nn.Linear(self.model_config.STATE_ENCODER.hidden_size, num_actions)

        self._init_layers()

        self.train()
        

    def pad_instructions(self, observations):
        instructions =[]
        for i in range(self.batch_size):
            instruction = observations['instruction'][i,:].unsqueeze(0)
            instructions.append(instruction.expand(int(observations['rgb'].shape[0]/self.batch_size), instruction.shape[1]).unsqueeze(0))    
        del instruction
        instructions = torch.cat(instructions, dim=0)
        instructions = instructions.view(-1, *instructions.size()[2:])
        return instructions.long()

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

    def forward_vnl(self, observations):
        # forwards for all timestep
        cnn_time = time.time()
        instruction_embedding = self.instruction_encoder(observations['instruction'].long())
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        instruction_embedding = instruction_embedding.expand(rgb_embedding.shape[0], instruction_embedding.shape[1])
        x = torch.cat([instruction_embedding, depth_embedding, rgb_embedding], dim=1)
        del instruction_embedding, depth_embedding, rgb_embedding
        # print("cnn high level time", time.time()-cnn_time)
        return x

    def forward(self, x, progress, rnn_hidden_states, prev_actions, masks, low_level_out):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        # observations, rnn_hidden_states, prev_actions, masks = batch
        # del batch

        # instructions = self.pad_instructions(observations)
        # del observations['instruction']
        # cnn_time = time.time()

        # instruction_embedding = self.instruction_encoder(observations['instruction'].long())
        # depth_embedding = self.depth_encoder(observations)
        # rgb_embedding = self.rgb_encoder(observations)
        # if self.model_config.ablate_instruction:
        #     instruction_embedding = instruction_embedding * 0
        # if self.model_config.ablate_depth:
        #     depth_embedding = depth_embedding * 0
        # if self.model_config.ablate_rgb:
        #     rgb_embedding = rgb_embedding * 0

        # instruction_embedding = instruction_embedding.expand(rgb_embedding.shape[0], instruction_embedding.shape[1])
        # x = torch.cat([instruction_embedding, depth_embedding, rgb_embedding], dim=1)
        # print("cnn high level time", time.time()-cnn_time)
        rest_time = time.time()

        # del instruction_embedding, depth_embedding, rgb_embedding
        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)
        masks = masks[:,0]

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1), progress, reduction="none"
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )
        detached_state_high = x.clone().detach()

        if low_level_out is not None:
            inter_module_out = self.inter_module_attn(x.unsqueeze(0), low_level_out, None, None)
            x = self.fc(torch.cat((x, inter_module_out.squeeze(0)), 1))

        x = self.linear(x)
        # print("rest time",time.time() - rest_time)
        return x, rnn_hidden_states, detached_state_high
