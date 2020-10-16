import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
# from habitat_baselines.rl.ppo.policy import Net
import time
from vlnce_baselines.common.utils import get_instruction_mask

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.language_encoder import LanguageEncoder
from vlnce_baselines.models.encoders.instruction_encoder import InstructionEncoder
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)

from vlnce_baselines.models.transformer.transformer import PositionEmbedding2DLearned
from vlnce_baselines.models.transformer.transformer import ImageEncoder_with_PosEncodings
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

        ## BERT Embedding
        self.embedding_layer = BertModel.from_pretrained('bert-base-uncased') 
        self.ins_fc =  nn.Linear(model_config.TRANSFORMER_INSTRUCTION_ENCODER.d_in, model_config.TRANSFORMER_INSTRUCTION_ENCODER.d_model)


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
                resnet_output=True,
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
                resnet_output=True,
            )

        self.visual_fc = nn.Linear(model_config.RGB_ENCODER.resnet_output_size+ model_config.DEPTH_ENCODER.output_size, 
                            model_config.IMAGE_CROSS_MODAL_ENCODER.d_model)

        N_steps = model_config.IMAGE_CROSS_MODAL_ENCODER.d_model // 2
        self.position_embedding_2d = PositionEmbedding2DLearned(N_steps)

        self.image_cm_encoder = ImageEncoder_with_PosEncodings(model_config.IMAGE_CROSS_MODAL_ENCODER)
        self.cross_pooler = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                          nn.Flatten())

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
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0


        # Fuse RGB-D
        rgb_d = torch.cat((rgb_embedding, depth_embedding), dim=1)
        rgb_d = F.relu(self.visual_fc(rgb_d.permute(0,2,3,1)))
        rgb_d = rgb_d.permute(0,3,1,2)


        instruction = observations["instruction"].long()
        lengths = (instruction != 0.0).long().sum(dim=1)


        # Using BERT Base Uncased for instruction embedding and avoiding any additional self attention blocks
        self.embedding_layer.eval()
        with torch.no_grad():
            embedded = self.embedding_layer(instruction)
            embedded = embedded[0]
        enc_mask = get_instruction_mask(embedded, lengths)
        ins_enc_out = self.ins_fc(embedded)

        lang_encoding = (ins_enc_out, enc_mask)        
        # instruction_embedding = self.instruction_encoder(observations['instruction'].long())
        # instruction_embedding = instruction_embedding.expand(rgb_embedding.shape[0], instruction_embedding.shape[1])
        # x = torch.cat([instruction_embedding, depth_embedding, rgb_embedding], dim=1)
        del depth_embedding, rgb_embedding, ins_enc_out, enc_mask
        # print("cnn high level time", time.time()-cnn_time)
        return rgb_d, lang_encoding

    def forward(self, x, lang_encoding, progress, rnn_hidden_states, prev_actions, masks, low_level_out):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        ins_embedding, ins_enc_mask = lang_encoding

        rgbd_pos_embed =  self.position_embedding_2d(x)
        x = x.flatten(2).permute(0, 2, 1) #B*49*resnet_output_size
        pos_embed_out = rgbd_pos_embed.flatten(2).permute(0, 2, 1) #B*49*resnet_output_size        
        x = self.image_cm_encoder(x, ins_embedding, None, ins_enc_mask, pos_embed_out)

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)
        masks = masks[:,0]
        x = self.cross_pooler(x.permute(0,2,1))
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
