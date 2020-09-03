import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.transformer.transformer import TransformerLanguageEncoder, PositionEmbedding2DLearned
from transformers import BertTokenizer, BertModel
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from vlnce_baselines.models.transformer.transformer import ImageEncoder_with_PosEncodings, SemMapEncoder_with_PosEncodings

from vlnce_baselines.models.transformer.transformer import ActionDecoderTransformer
from vlnce_baselines.models.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from vlnce_baselines.models.policy import BaseTransformerPolicy
from vlnce_baselines.common.utils import get_transformer_mask, get_instruction_mask


class TransformerSemPolicy(BaseTransformerPolicy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config, batch_size, gpus
    ):
        super().__init__(
            TransformerEncDec(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
                batch_size=batch_size,
                gpus = gpus
            ),
            action_space.n,
        )

class TransformerEncDec(Net):
    r"""A Seq to Seq Polocy based entirely on Attention and contains no recurrent module.

    Modules:
        Trasformer based Instruction encoder
        transformer based cross modal Image Encoder
        RGB encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions, batch_size, gpus):
        super().__init__()
        self.model_config = model_config
        self.downsample_size = self.model_config.SEM_MAP_TRANSFORMER.downsample_size

        self.device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.batch_size = batch_size

        ## BERT Embedding
        self.embedding_layer = BertModel.from_pretrained('bert-base-uncased') 

        self.ins_fc =  nn.Linear(model_config.TRANSFORMER_INSTRUCTION_ENCODER.d_in, model_config.TRANSFORMER_INSTRUCTION_ENCODER.d_model)

        # Init the instruction encoder
        # self.instruction_encoder = TransformerLanguageEncoder(model_config.TRANSFORMER_INSTRUCTION_ENCODER)

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
                self.device, 
                resnet_output=True,
            )

        self.model_config.defrost()
        self.model_config.SEM_MAP_TRANSFORMER.map_size = int(observation_space.spaces["ego_sem_map"].high.max() + 1)
        self.model_config.SEM_MAP_TRANSFORMER.max_position_embeddings=observation_space.spaces["ego_sem_map"].shape[0] // 2
        self.model_config.freeze()

        self.sem_map_encoder = SemMapEncoder_with_PosEncodings(model_config.SEM_MAP_TRANSFORMER)

        self.sem_pooler = nn.AdaptiveAvgPool1d(16)

        self.prev_action_embedding = nn.Embedding(num_actions+1, 32, padding_idx= 1)
        
        N_steps = model_config.IMAGE_CROSS_MODAL_ENCODER.d_model // 2
        self.position_embedding_2d = PositionEmbedding2DLearned(N_steps)

        self.image_cm_encoder = ImageEncoder_with_PosEncodings(model_config.IMAGE_CROSS_MODAL_ENCODER)

        self.action_decoder = ActionDecoderTransformer(model_config.ACTION_DECODER_TRANFORMER)

        self.progress_monitor = nn.Linear(
            self.model_config.TRANSFORMER.output_size, 1
        )

        self.visual_fc = nn.Linear(model_config.RGB_ENCODER.resnet_output_size+ model_config.DEPTH_ENCODER.output_size, 
                            model_config.IMAGE_CROSS_MODAL_ENCODER.d_model)

        self.visual_sem_fc = nn.Linear(model_config.IMAGE_CROSS_MODAL_ENCODER.d_model*2,  model_config.IMAGE_CROSS_MODAL_ENCODER.d_model)
        self._init_layers()
        self.train()

    @property
    def output_size(self):
        return self.model_config.TRANSFORMER.output_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        pass

    def _init_layers(self):
        nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
        nn.init.constant_(self.progress_monitor.bias, 0)

    def single_forward(self, visual_embeddings, lang_encoding, progress, prev_actions, prev_actions_list, masks, prev_action_pad_mask=None):
        r"""
        ins_enc_out: [batch_size x MAX_LEN, TRANSFORMER_INSTRUCTION_ENCODER.d_model]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_embedding, depth_embedding, sem_map = visual_embeddings
        ins_enc_out, enc_mask = lang_encoding

        del visual_embeddings
        del lang_encoding

        ## Fuse RGB-D
        rgb_d = torch.cat((rgb_embedding, depth_embedding), dim=1)
        rgb_d = F.relu(self.visual_fc(rgb_d.permute(0,2,3,1)))
        rgb_d = rgb_d.permute(0,3,1,2)

        rgbd_pos_embed =  self.position_embedding_2d(rgb_d)
        rgbd_out = rgb_d.flatten(2).permute(0, 2, 1) #B*49*resnet_output_size
        pos_embed_out = rgbd_pos_embed.flatten(2).permute(0, 2, 1) #B*49*resnet_output_size

        visual_enc_out = self.image_cm_encoder(rgbd_out, ins_enc_out, None, enc_mask, pos_embed_out)
        del rgb_d, rgbd_out, rgb_embedding,depth_embedding, rgbd_pos_embed

        sem_map = sem_map.unsqueeze(1)
        sem_map = F.interpolate(sem_map, size=(self.downsample_size, self.downsample_size), mode='nearest') #Downsample map
        sem_map = sem_map.squeeze(1)
        sem_enc_out = self.sem_map_encoder(sem_map, ins_enc_out, None, enc_mask)
        sem_enc_out = self.sem_pooler(sem_enc_out.permute(0,2,1)).permute(0,2,1)

        visual_semantic_enc = torch.cat((visual_enc_out, sem_enc_out), dim=-1)
        visual_semantic_enc = F.relu(self.visual_sem_fc(visual_semantic_enc))

        del sem_enc_out, visual_enc_out, sem_map

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        if self.model_config.TRANSFORMER.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long()
            )

        if not (prev_actions_list is None):
            prev_actions_list.append(prev_actions_embedding)
            prev_action_seq = torch.stack(prev_actions_list).transpose(0,1)
        else:
            prev_action_seq = prev_actions_embedding

        del prev_actions_embedding
        out = self.action_decoder(prev_action_seq, ins_enc_out, visual_semantic_enc, enc_att_mask_w=enc_mask, 
                                    enc_att_mask_i=None, device=self.device, pos_embed= pos_embed_out)

        if not (prev_actions_list is None):
            return out, prev_actions_list
        else:
            return out 

    def seq_forward(self, visual_embeddings, lang_encoding, progress, prev_actions, masks):
        rgb_embedding, depth_embedding, sem_map = visual_embeddings
        rgb_out_dim = rgb_embedding.shape[3]
        depth_out_dim = depth_embedding.shape[3]
        
        max_len          = int(prev_actions.size(0)/self.batch_size)
        prev_actions     = prev_actions.view(self.batch_size,max_len)
        masks            = masks.view(self.batch_size,max_len)
        rgb_embedding    = rgb_embedding.view(self.batch_size, max_len, -1, rgb_out_dim, rgb_out_dim)
        depth_embedding  = depth_embedding.view(self.batch_size, max_len, -1, depth_out_dim, depth_out_dim)
        sem_map          = sem_map.view(self.batch_size, max_len,*sem_map.size()[1:])

        output   = []
        for i in range(max_len):
            prev_actions_single       = prev_actions[:,:i+1]
            masks_single              = masks[:,:i+1]
            rgb_single_embed          = rgb_embedding[:,i]
            depth_single_embed        = depth_embedding[:,i]
            sem_single_map            = sem_map[:,i]
            prev_action_pad_mask      = (((prev_actions_single+1)*masks_single)!= 1).unsqueeze(1).unsqueeze(2)
            visual_embeddings         = (rgb_single_embed, depth_single_embed, sem_single_map)

            del rgb_single_embed, sem_single_map, depth_single_embed
            out = self.single_forward(visual_embeddings, lang_encoding, progress, prev_actions_single, None, masks_single, prev_action_pad_mask = prev_action_pad_mask)
            out = out.unsqueeze(1)
            output.append(out)
            del out

        output = torch.cat(output,dim=1) 
        output = output.view(self.batch_size*max_len, -1)         
        return output


    def forward(self, observations, prev_actions_list, prev_actions, masks):
        progress = observations['progress']
        instruction = observations["instruction"].long()
        lengths = (instruction != 0.0).long().sum(dim=1)

        self.embedding_layer.eval()
        with torch.no_grad():
            embedded = self.embedding_layer(instruction)
            embedded = embedded[0]
        #Value = (Instruction Embedded, Positional Mask), AttnMask = Self Attention Mask, enc_mask = Encoder mask for cross attention
        # value, attn_mask, enc_mask = get_transformer_mask(embedded, lengths)
        enc_mask = get_instruction_mask(embedded, lengths)

        # Using BERT Base Uncased for instruction embedding and avoiding any additional self attention blocks
        ins_enc_out = self.ins_fc(embedded)

        # #ins_enc_out: B*max_len*hidden_dim (hidden_dim = resnet_output_size)
        # ins_enc_out = self.instruction_encoder(value, attention_mask=attn_mask, attention_weights=None, device = self.device)
         
                
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations) #B*resnet_output_size*7*7

        batch_dim = observations['progress'].shape[0]
        num_process_dim = observations['instruction'].shape[0]
        sem_map = observations['ego_sem_map']

        del observations
        visual_embeddings = (rgb_embedding, depth_embedding, sem_map)

        lang_encoding = (ins_enc_out, enc_mask)
        del ins_enc_out, enc_mask

        if batch_dim == num_process_dim:
            prev_actions = prev_actions.view(-1)
            masks = masks.view(-1)
            output, prev_actions_list = self.single_forward(visual_embeddings, lang_encoding, progress, prev_actions,prev_actions_list, masks)
            if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
                progress_hat = torch.tanh(self.progress_monitor(output))
                progress_loss = F.mse_loss(
                    progress_hat.squeeze(1), progress, reduction="none"
                )
                AuxLosses.register_loss(
                    "progress_monitor",
                    progress_loss,
                    self.model_config.PROGRESS_MONITOR.alpha,
                )
        else:
            output = self.seq_forward(visual_embeddings, lang_encoding, progress, prev_actions, masks)
            if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
                progress_hat = torch.tanh(self.progress_monitor(output))
                progress_loss = F.mse_loss(
                    progress_hat.squeeze(1), progress, reduction="none"
                )
                AuxLosses.register_loss(
                    "progress_monitor",
                    progress_loss,
                    self.model_config.PROGRESS_MONITOR.alpha,
                )
        return output, prev_actions_list