import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from robo_vln_baselines.common.utils import sinusoid_encoding_table
from typing import Optional
from torch import Tensor

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed forward layer
    """

    def __init__(self, d_model=512, d_ff=2048, dropout=.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        # HERE BERT Style F.gelu is used
        pwff = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        pwff = self.dropout(pwff)
        out = self.layer_norm(input + pwff)
        return out
        

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights(gain=1.0)

    def init_weights(self, gain=1.0):
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_o.weight, gain=gain)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            # TODO a bit different from Herdade et al. 2019
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, 0)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(MultiHeadAttention, self).__init__()

        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.attention(queries, keys, values, attention_mask, attention_weights)
        att = self.dropout(att)
        return self.layer_norm(queries + att)
        # # return queries + att
        # return att

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff

class BaseEncoder(nn.Module):
    def __init__(self, config):
        super(BaseEncoder, self).__init__()

        self.d_model = config.d_model
        self.d_att = int(self.d_model/config.h)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_att, self.d_att, config.h, config.d_ff, config.dropout) for _ in range(config.N)])
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_mask, attention_weights=None):
        # input (b_s, seq_len, d_in)
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)

        return out

class TransformerLanguageEncoder(BaseEncoder):
    def __init__(self, config):
        super(TransformerLanguageEncoder, self).__init__(config)
        self.fc = nn.Linear(config.d_in, self.d_model, bias=True)

    def forward(self, input, attention_mask, attention_weights=None):
        # input (b_s, seq_len, d_in)
        # data, mask = input

        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        dev = out.get_device()

        #TODO:  Add learned positional encoding embedding layer rather than sine+cosine embedding: 
        # Reference: https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb

        pe = sinusoid_encoding_table(out.shape[1], out.shape[2])
        pe = pe.expand(out.shape[0], pe.shape[0], pe.shape[1]).to(dev)
        out = out + pe.masked_fill(mask, 0)

        out = super(TransformerLanguageEncoder, self).forward(out, attention_mask, attention_weights=attention_weights)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, pooler=False):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.pooler=pooler
        if self.pooler:
            self.pooler = nn.AdaptiveAvgPool1d(100)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, input, enc_output, mask_self_att, mask_enc_att, pos_embed=None):
        if pos_embed is not None:
            input =  self.with_pos_embed(input, pos_embed)
        self_att = self.self_att(input, input, input, mask_self_att)
        if pos_embed is not None:
            self_att =  self.with_pos_embed(self_att, pos_embed)
        if self.pooler:
            self_att = self.pooler(self_att.permute(0,2,1)).permute(0,2,1)
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        ff = self.pwff(enc_att)
        return ff

class InterModuleAttnLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, pooler=False):
        super(InterModuleAttnLayer, self).__init__()
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, input_1, input_2, mask_self_att, mask_enc_att, pos_embed=None):
        enc_att = self.enc_att(input_1, input_2, input_2, mask_enc_att)
        ff = self.pwff(enc_att)
        return ff
        # return enc_att

class InterModuleAttnDecoder(nn.Module):
    def __init__(self, config):
        super(InterModuleAttnDecoder, self).__init__()
        self.d_model = config.d_model
        self.d_att = int(self.d_model/config.h)
        self.layers = nn.ModuleList([InterModuleAttnLayer(self.d_model, self.d_att, self.d_att, config.h, config.d_ff, config.dropout) for _ in range(config.N)])
        self.fc = nn.Linear(config.in_features, self.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, input_2, self_att_mask, enc_att_mask):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        input_2 = F.relu(self.fc(input_2))
        input_2 = self.dropout(input_2)
        input_2 = self.layer_norm(input_2)

        ##TODO: Try making Image encoder same as DETR Image encoder: https://github.com/facebookresearch/detr/blob/ae03a2d6e52a9ec1b67f85437d0a275c5abbe9ac/models/transformer.py#L149
        ## Add position embed to only query and key vector and not value

        for l in self.layers:
            out = l(out, input_2, self_att_mask, enc_att_mask)
        return out


class Visual_Ling_Attn(nn.Module):
    def __init__(self, config):
        super(Visual_Ling_Attn, self).__init__()
        self.d_model = config.d_model
        self.d_att = int(self.d_model/config.h)
        self.layers = nn.ModuleList([InterModuleAttnLayer(self.d_model, self.d_att, self.d_att, config.h, config.d_ff, config.dropout) for _ in range(config.N)])
        self.vis_fc = nn.Linear(config.vis_in_features, self.d_model)
        self.ins_fc = nn.Linear(config.ins_in_features, self.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, input_2, self_att_mask, enc_att_mask):
        out = F.relu(self.vis_fc(input_2))
        out = self.dropout(out)
        out = self.layer_norm(out)

        input = F.relu(self.ins_fc(input))
        input = self.dropout(input)
        input = self.layer_norm(input)
        
        dev = input.get_device()
        pe = sinusoid_encoding_table(input.shape[1], input.shape[2])
        pe = pe.expand(input.shape[0], pe.shape[0], pe.shape[1]).to(dev)
        input = input + pe

        ##TODO: Try making Image encoder same as DETR Image encoder: https://github.com/facebookresearch/detr/blob/ae03a2d6e52a9ec1b67f85437d0a275c5abbe9ac/models/transformer.py#L149
        ## Add position embed to only query and key vector and not value

        for l in self.layers:
            out = l(input, out, self_att_mask, enc_att_mask)
        return out

class ImageCrossModalEncoder(nn.Module):
    def __init__(self, config):
        super(ImageCrossModalEncoder, self).__init__()
        self.d_model = config.d_model
        self.d_att = int(self.d_model/config.h)
        self.layers = nn.ModuleList([DecoderLayer(self.d_model, self.d_att, config.d_att, config.h, config.d_ff, config.dropout) for _ in range(config.N)])
        self.fc = nn.Linear(config.in_features, self.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, enc_output, self_att_mask, enc_att_mask):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        ##TODO: Try making Image encoder same as DETR Image encoder: https://github.com/facebookresearch/detr/blob/ae03a2d6e52a9ec1b67f85437d0a275c5abbe9ac/models/transformer.py#L149
        ## Add position embed to only query and key vector and not value

        for l in self.layers:
            out = l(out, enc_output, self_att_mask, enc_att_mask)
        return out

class ImageEncoder_with_PosEncodings(nn.Module):
    def __init__(self, config):
        super(ImageEncoder_with_PosEncodings, self).__init__()
        self.d_model = config.d_model
        self.d_att = int(self.d_model/config.h)
        self.layers = nn.ModuleList([DecoderLayer(self.d_model, self.d_att, self.d_att, config.h, config.d_ff, config.dropout, False) for _ in range(config.N)])
        self.fc = nn.Linear(config.d_in, self.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, enc_output, self_att_mask, enc_att_mask, pos_embed):
        # out = F.relu(self.fc(input))
        out = self.dropout(input)
        out = self.layer_norm(out)

        ##TODO: Try making Image encoder same as DETR Image encoder: https://github.com/facebookresearch/detr/blob/ae03a2d6e52a9ec1b67f85437d0a275c5abbe9ac/models/transformer.py#L149
        ## Add position embed to only query and key vector and not value
        
        for l in self.layers:
            out = l(out, enc_output, self_att_mask, enc_att_mask, pos_embed)

        return out

class ImagePlainEncoder(BaseEncoder):
    def __init__(self, N, d_in=300, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(ImagePlainEncoder, self).__init__(N, d_model, d_k, d_v, h, d_ff, dropout)
        self.fc = nn.Linear(d_in, self.d_model)

    def forward(self, input, attention_mask, attention_weights=None):
        # input (b_s, seq_len, d_in)
        data = input

        out = F.relu(self.fc(data))
        out = self.dropout(out)
        out = self.layer_norm(out)

        out = super(ImagePlainEncoder, self).forward(out, attention_mask, attention_weights)
        return out

class PositionEmbedding2DLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        x = tensor
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)

        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
