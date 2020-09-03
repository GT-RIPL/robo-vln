import gzip
import json

import torch
import torch.nn as nn
from habitat import Config, logger
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel


class LanguageEncoder(nn.Module):
    def __init__(self, config: Config, device):
        r"""An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                vocab_size: number of words in the vocabulary
                embedding_size: The dimension of each embedding vector
                use_pretrained_embeddings:
                embedding_file:
                fine_tune_embeddings:
                dataset_vocab:
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: Whether or not to return just the final state
        """
        super().__init__()

        self.config = config
        self.device = device
        self.num_layers = config.num_layers
        self.dropout_ratio = config.dropout_ratio
        self.drop = nn.Dropout(p=self.dropout_ratio)

        if self.config.use_pretrained_embeddings:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=self._load_embeddings(),
                freeze=not self.config.fine_tune_embeddings,
            )
        elif self.config.is_bert:
            self.embedding_layer = BertModel.from_pretrained('bert-base-uncased') 
        else:  # each embedding initialized to sampled Gaussian
            self.embedding_layer = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_size,
                padding_idx=0,
            )

        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.bidir = config.bidirectional
        self.encoder_rnn = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=self.bidir,
            dropout=self.dropout_ratio,
            batch_first= True,
        )
        self.num_directions = 2 if self.bidir else 1
        self.encoder2decoder = nn.Linear(config.hidden_size * self.num_directions,
            config.hidden_size * self.num_directions
        )
        self.final_state_only = config.final_state_only

    @property
    def output_size(self):
        return self.config.hidden_size * (2 if self.bidir else 1)

    def _init_state(self, batch_size):
        
        h0 = Variable(torch.zeros(
            self.num_layers,
            batch_size,
            self.config.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers,
            batch_size,
            self.config.hidden_size
        ), requires_grad=False)

        return h0.to(self.device), c0.to(self.device)

    def _load_embeddings(self):
        """ Loads word embeddings from a pretrained embeddings file.

        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged:
            https://groups.google.com/forum/#!searchin/globalvectors/unk|sort:date/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ

        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations):
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        instruction = observations["instruction"].long()
        # instruction = observations["instruction_batch"]

        lengths = (instruction != 0.0).long().sum(dim=1)

        if self.config.is_bert:
            self.embedding_layer.eval()
            with torch.no_grad():
                embedded = self.embedding_layer(instruction)
                embedded = embedded[0]
        else:
            embedded = self.embedding_layer(instruction)
        # print("Embedded shape:",embedded.shape)
        # batch_size = observations["instruction"].shape[0]
        embedded = self.drop(embedded)
        batch_size = observations["instruction"].shape[0]
        h0, c0 = self._init_state(batch_size)

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        output, hidden = self.encoder_rnn(packed_seq, (h0,c0))

        # if self.config.rnn_type == "LSTM":
        #     final_state = final_state[0]

        if self.final_state_only:
            output = hidden[0].squeeze(0)
            return output
        else:
            # return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
            #     0
            # ].permute(0, 2, 1), hidden
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        # print("hidden", hidden[0].shape)

        h_t = torch.tanh(self.encoder2decoder(hidden[0][-1]))
        h_t = h_t.unsqueeze(0)
        c_t = hidden[1]
        hidden = (h_t, c_t)
        return output, hidden
