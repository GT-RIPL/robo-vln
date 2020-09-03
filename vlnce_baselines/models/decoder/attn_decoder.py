import torch.nn as nn

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SoftAttn(nn.Module):
    def __init__(self, method, hidden_size):
        super(SoftAttn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size, bias=False)

    def forward(self, hidden, encoder_outputs, decoder_max_len,mask, device):

        batch_size, output_len, dimensions = hidden.size()
        max_len = encoder_outputs.size(1)

        if self.method == "general":
            hidden = hidden.reshape(batch_size * output_len, dimensions)
            hidden = self.attn(hidden)
            hidden = hidden.reshape(batch_size, output_len, dimensions)
        attention_scores = torch.bmm(hidden, encoder_outputs.transpose(1, 2).contiguous())
        attention_scores.data.masked_fill_(mask == 0, -float('inf'))
        # Mask layers with
        attention_weights = F.softmax(attention_scores, dim=-1)
        # attention_weights = attention_weights.view(batch_size, output_len, max_len)

        # Create variable to store attention energies
        # attn_energies = Variable(torch.zeros(this_batch_size, decoder_max_len, max_len)) # B x S
        # attn_energies_per_decoder_input = torch.zeros(batch_size, max_len) # B x S
        # attn_energies=[]

        # attn_energies = attn_energies.to(device)

        # For each batch of encoder outputs
        # for c in range(decoder_max_len):
        #     for b in range(this_batch_size):
        #         # Calculate energy for each encoder output
        #         for i in range(max_len):
        #             attn_energies_per_decoder_input[b,i] = self.score(hidden[b, c], encoder_outputs[b, i].unsqueeze(0))
        #     attn_energies.append(F.softmax(attn_energies_per_decoder_input, dim=1).unsqueeze(1))
        # # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        # attn_energies = torch.cat(attn_energies, dim=1)
        # attn_energies = Variable(attn_energies).to(device)
        return attention_weights.to(device)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = torch.dot(hidden.contiguous().view(-1), energy.contiguous().view(-1))
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.contiguous().view(-1), energy.contiguous().view(-1))
            return energy

class Attn_Decoder(nn.Module):
    def __init__(
        self,
        attn_model,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
        dropout_ratio = 0.25,
    ):
        r"""An RNN for encoding the state in RL.
        Supports masking the hidden state during various timesteps in the forward lass
        Args:
            input_size: The input size of the RNN
            hidden_size: The hidden size
            num_layers: The number of recurrent layers
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """

        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first = True,
        )
        if attn_model != 'none':
            self.attn = SoftAttn(attn_model, hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.drop = nn.Dropout(p=dropout_ratio)

        self.layer_init()

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (
            2 if "LSTM" in self._rnn_type else 1
        )

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat(
                [hidden_states[0], hidden_states[1]], dim=0
            )

        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )

        return hidden_states

    def _mask_hidden(self, hidden_states, masks):
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states

        return hidden_states

    def _create_mask(self,batchsize, max_length, length,device):
        """Given the length create a mask given a padded tensor"""
        tensor_mask = torch.zeros(batchsize, max_length, dtype = torch.bool)
        length = length.to(dtype=torch.long)
        for idx, row in enumerate(tensor_mask):
            row[:length[idx]] = 1
        tensor_mask.unsqueeze_(-1)
        return tensor_mask.to(device)

    def single_forward(self, x, hidden_states, encoder_output, lengths, device):
        r"""Forward for a non-sequence input
        """
        # hidden_states = self._unpack_hidden(hidden_states)

        batch_size = x.size(0)
        x= x.unsqueeze(1)
        x = self.drop(x)
        output, hidden_states = self.rnn(x, hidden_states)
        # print(output.shape)

        mask = self._create_mask(encoder_output.shape[0], encoder_output.shape[1], lengths, device)
        mask = mask.permute(0,2,1)
        decoder_max_len = x.shape[1]
        attn_weights = self.attn(output, encoder_output,decoder_max_len,mask,device)
        context = attn_weights.bmm(encoder_output)
        concat_input = torch.cat((output, context), 2)
        concat_input = concat_input.contiguous().view(batch_size, -1)  # flatten
        concat_output = torch.tanh(self.concat(concat_input))
        hidden_states = self._pack_hidden(hidden_states)

        # x, hidden_states = self.rnn(
        #     x.unsqueeze(0),
        #     self._mask_hidden(hidden_states, masks.unsqueeze(0)),
        # )
        # x = x.squeeze(0)
        # hidden_states = self._pack_hidden(hidden_states)
                # # x is a (T, N, -1) tensor
        return concat_output, hidden_states, attn_weights

    def seq_forward(self, x, hidden_states, encoder_output, lengths, device):
        r"""Forward for a sequence of length T
        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        batch_size = hidden_states[0].size(1)

        N = int(x.size(0)/batch_size)

        # unflatten
        x = x.view(batch_size, N, -1)
        x = self.drop(x)
        output, hidden_states = self.rnn(x, hidden_states)

        mask = self._create_mask(encoder_output.shape[0], encoder_output.shape[1], lengths, device)
        mask = mask.permute(0,2,1)
        decoder_max_len = output.shape[1]
        mask = mask.expand(encoder_output.shape[0], decoder_max_len, encoder_output.shape[1])

        attn_weights = self.attn(output, encoder_output,decoder_max_len,mask,device)
        context = attn_weights.bmm(encoder_output)
        concat_input = torch.cat((output, context), 2)
        concat_input = concat_input.contiguous().view(batch_size*N, -1)  # flatten
        concat_output = torch.tanh(self.concat(concat_input))

        hidden_states = self._pack_hidden(hidden_states)
        return concat_output, hidden_states, attn_weights

    def forward(self, x, hidden_states, encoder_output, lengths, device):
        # print(x.size())
        # print(hidden_states[0].size())
        if x.size(0) == hidden_states[0].size(1):
            return self.single_forward(x, hidden_states, encoder_output, lengths, device)
        else:
            return self.seq_forward(x, hidden_states, encoder_output, lengths, device)

class Attn_DecoderSequence(nn.Module):
    def __init__(
        self,
        attn_model,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
        dropout_ratio = 0.25,
    ):
        r"""An RNN for encoding the state in RL.
        Supports masking the hidden state during various timesteps in the forward lass
        Args:
            input_size: The input size of the RNN
            hidden_size: The hidden size
            num_layers: The number of recurrent layers
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """

        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first = True,
        )
        if attn_model != 'none':
            self.attn = SoftAttn(attn_model, hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.drop = nn.Dropout(p=dropout_ratio)

        self.layer_init()

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (
            2 if "LSTM" in self._rnn_type else 1
        )

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat(
                [hidden_states[0], hidden_states[1]], dim=0
            )

        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )

        return hidden_states

    def _mask_hidden(self, decoder_hidden, encoder_hidden, masks):
        if isinstance(decoder_hidden, tuple):
            done_masks = 1-masks
            hidden_states = tuple(v * masks + w*done_masks for v,w in 
            zip(decoder_hidden, encoder_hidden))
        return hidden_states

    def _create_mask(self,batchsize, max_length, length,device):
        """Given the length create a mask given a padded tensor"""
        tensor_mask = torch.zeros(batchsize, max_length, dtype = torch.bool)
        length = length.to(dtype=torch.long)
        for idx, row in enumerate(tensor_mask):
            row[:length[idx]] = 1
        tensor_mask.unsqueeze_(-1)
        # print(tensor_mask)
        return tensor_mask.to(device)

    def single_forward(self, x, hidden_states, encoder_hidden, encoder_output, masks, lengths,device, pack_hidden=False):
        r"""Forward for a non-sequence input
        """
        if not isinstance(hidden_states, tuple):
            hidden_states = self._unpack_hidden(hidden_states)
        
        hidden_states = self._mask_hidden(hidden_states, encoder_hidden, masks.unsqueeze(0))
        batch_size = x.size(0)
        x= x.unsqueeze(1)
        x = self.drop(x)
        output, hidden_states = self.rnn(x, hidden_states)

        mask = self._create_mask(encoder_output.shape[0], encoder_output.shape[1], lengths, device)
        mask = mask.permute(0,2,1)
        decoder_max_len = x.shape[1]
        attn_weights = self.attn(output, encoder_output,decoder_max_len,mask,device)
        context = attn_weights.bmm(encoder_output)
        concat_input = torch.cat((output, context), 2)
        concat_input = concat_input.contiguous().view(batch_size, -1)  # flatten
        concat_output = torch.tanh(self.concat(concat_input))

        if pack_hidden:
            hidden_states = self._pack_hidden(hidden_states)

        return concat_output, hidden_states, attn_weights

    def seq_forward(self, x, hidden_states, encoder_hidden, encoder_output, masks, lengths, device):
        r"""Forward for a sequence of length T
        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        batch_size = hidden_states[0].size(1)

        masks = torch.ones(batch_size, 1, device=device)
        N = int(x.size(0)/batch_size)
        x = x.view(batch_size, N, -1)
        x = self.drop(x)
        decoder_max_len = x.shape[1]
        output = []
        all_attn_weights = []
        for i in range(decoder_max_len):
            visual_embed = x[:,i,:]
            concat_output, hidden_states, attn_weights = self.single_forward(
                visual_embed, hidden_states, encoder_hidden, encoder_output, masks, lengths, device, pack_hidden = False)
            concat_output = concat_output.unsqueeze(1)
            output.append(concat_output)
            all_attn_weights.append(attn_weights)

        output = torch.cat(output, dim=1)
        all_attn_weights = torch.cat(all_attn_weights, dim=1)

        hidden_states = self._pack_hidden(hidden_states)
        output = output.contiguous().view(batch_size*N, -1)  # flatten

        return output, hidden_states, all_attn_weights

    def forward(self, x, hidden_states, encoder_hidden, encoder_output, masks, lengths, device):
        #hidden state has to be a tuple here
        if isinstance(hidden_states, tuple):
            if x.size(0) == hidden_states[0].size(1):
                return self.single_forward(x, hidden_states, encoder_hidden, encoder_output, masks, lengths, device, pack_hidden=True)
            else:
                return self.seq_forward(x, hidden_states, encoder_hidden, encoder_output, masks, lengths, device)
        else: 
            if x.size(0) == hidden_states.size(1):
                return self.single_forward(x, hidden_states, encoder_hidden, encoder_output, masks, lengths, device, pack_hidden=True)
            else:
                return self.seq_forward(x, hidden_states, encoder_hidden, encoder_output, masks, lengths, device)
