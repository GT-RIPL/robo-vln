import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
import numpy as np
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
import time
from robo_vln_baselines.common.utils import get_instruction_mask

from robo_vln_baselines.common.aux_losses import AuxLosses
from robo_vln_baselines.models.encoders.language_encoder import LanguageEncoder
from robo_vln_baselines.models.encoders.instruction_encoder import InstructionEncoder
from robo_vln_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from robo_vln_baselines.models.transformer.transformer import TransformerLanguageEncoder
from transformers import BertModel
from robo_vln_baselines.common.utils import get_transformer_mask

from robo_vln_baselines.models.transformer.transformer import PositionEmbedding2DLearned
from robo_vln_baselines.models.transformer.transformer import ImageEncoder_with_PosEncodings
from robo_vln_baselines.models.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from robo_vln_baselines.models.transformer.transformer import Visual_Ling_Attn, InterModuleAttnDecoder

class Seq2Seq_HighLevel_CMA(nn.Module):
    r"""A baseline high-level Cross-Modal Decoder 
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
        

        ## BERT Embedding
        self.embedding_layer = BertModel.from_pretrained('bert-base-uncased') 
        self.ins_fc =  nn.Linear(model_config.TRANSFORMER_INSTRUCTION_ENCODER.d_in, model_config.TRANSFORMER_INSTRUCTION_ENCODER.d_model)
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
                spatial_output=True,
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
                device,
                spatial_output=True,
            )

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            model_config.VISUAL_LING_ATTN.vis_in_features,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            model_config.VISUAL_LING_ATTN.vis_in_features,
            1,
        )
        self.image_cm_encoder = Visual_Ling_Attn(model_config.VISUAL_LING_ATTN)
        self.cross_pooler = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                          nn.Flatten())

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        rnn_input_size = (
            self.model_config.IMAGE_CROSS_MODAL_ENCODER.d_model*2
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
        self.linear = nn.Linear(self.model_config.STATE_ENCODER.hidden_size, num_actions)

        self._init_layers()
        

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

    def forward(self, batch):
        r"""
        (observations, rnn_hidden_states, prev_actions, masks) = batch
        """

        observations, rnn_hidden_states, prev_actions, masks = batch
        del batch

        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)
        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0
        instruction = observations["instruction"].long()
        instruction = instruction.expand(rgb_embedding.shape[0], observations['instruction'].shape[1])

        self.embedding_layer.eval()
        with torch.no_grad():
            embedded = self.embedding_layer(instruction)
            embedded = embedded[0]
        del observations['instruction']

        rgb_spatial = self.rgb_kv(rgb_embedding)
        depth_spatial = self.depth_kv(depth_embedding)
        ins_rgb_att = self.image_cm_encoder(embedded, rgb_spatial.permute(0,2,1), None, None)
        ins_depth_att = self.image_cm_encoder(embedded, depth_spatial.permute(0,2,1),None, None)

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)
        masks = masks[:,0]
        ins_rgb_att = self.cross_pooler(ins_rgb_att.permute(0,2,1))
        ins_depth_att = self.cross_pooler(ins_depth_att.permute(0,2,1))


        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)
        x = torch.cat((rgb_in, depth_in, ins_rgb_att, ins_depth_att), dim=1)

        del rgb_embedding, depth_embedding, rgb_in, depth_in, ins_rgb_att, ins_depth_att

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

        x = self.linear(x)
        return x, rnn_hidden_states
