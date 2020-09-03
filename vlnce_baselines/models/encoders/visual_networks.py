import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from vlnce_baselines.models.encoders.my_simple_cnn import SimpleCNN
from vlnce_baselines.models.encoders.bert_models import BertConfig, BertModel
# from scripts.training.graph_models import SceneGraphModel, SceneGTNModel


class CNNMapEmbedding(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_size):
        super().__init__()
        self.semantic_embedding = nn.Embedding(
            vocab_dim + 1, embedding_dim, padding_idx=0)

        self.conv_embedding = nn.Conv2d(embedding_dim, hidden_size, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x will be of shape B*40*40
        x = self.semantic_embedding(x)  # B*40*40*128
        x = x.permute(0, 3, 1, 2)  # B*128*40*40
        x = F.normalize(x)
        x = self.conv_embedding(x)  # B*512*40*40
        x = self.flatten(self.avg_pool(x))  # B*512
        return x


class Attention(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.semantic_embedding = nn.Embedding(
            vocab_dim + 1, embedding_dim, padding_idx=0)
        # convert B*128*40*40 -> B*512*40*40
        self.query_conv = nn.Conv2d(embedding_dim, hidden_size, 1, 1)
        self.value_conv = nn.Conv2d(embedding_dim, hidden_size, 1, 1)
        # convert B*512 -> to something that can be multipled with it
        self.key_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.conv_embedding = nn.Conv2d(hidden_size, hidden_size, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rgb_embedding):
        mask = x == 0  # B*W*H
        rgb_embedding = rgb_embedding.view(-1, self.hidden_size, 1, 1)
        x = self.semantic_embedding(x)
        x = x.permute(0, 3, 1, 2)
        x = F.normalize(x)  # B X 128  X W X H
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N X(C)
        # .view(m_batchsize,-1,width*height) # B X C x (1)
        proj_key = self.key_conv(rgb_embedding).view(m_batchsize, -1, 1)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X N
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        energy = energy / np.sqrt(width * height)
        energy = energy.view(-1, width, height)
        # replace mask zeros with -inf so that softmax will be zeros
        energy = energy.masked_fill(mask, -1e9)

        # BX (N) X (N)  # Question: softmax axis
        attention = self.softmax(energy)
        # B X C X 1 (B x C X N * B X N X1)
        out = torch.bmm(proj_value, attention.view(-1, width * height, 1))
        x = out

        x = self.conv_embedding(
            x.view(-1, self.hidden_size, 1, 1))  # B X C X 1
        x = self.flatten(self.avg_pool(x))  # B X C
        return x


class RGBDSimpleCNN(nn.Module):
    def __init__(self, observation_space, config):
        super().__init__()
        self._hidden_size = config.hidden_size
        self.spaces = []
        self.shapes = {}
        if "rgb" in observation_space.spaces:
            self.spaces.append("rgb")
            self.shapes['rgb'] = observation_space.spaces['rgb'].high.max()

        if "depth" in observation_space.spaces:
            self.spaces.append("depth")
            self.shapes['depth'] = observation_space.spaces['depth'].high.max()
        self._in_features = 512
        self.cnn = SimpleCNN(observation_space, self.spaces, self._in_features)
        self.fc_before_GRU = nn.Linear(self._in_features, self._hidden_size)

    def imagenet_preprocess(self, img):
        return img  # output

    @property
    def is_blind(self):
        return self.cnn.is_blind

    def visual_forward(self, observations):
        cnn_input = []
        for modality in self.spaces:
            observation = observations[modality]
            observation = observation / self.shapes[modality]  # normalize RGB
            observation = self.imagenet_preprocess(observation)
            observation = observation.permute(0, 3, 1, 2)
            cnn_input.append(observation)

        cnn_input = torch.cat(cnn_input, dim=1)
        outputs = self.cnn(cnn_input)
        return outputs

    def forward(self, observations):
        outputs = self.visual_forward(observations)
        final_outputs = self.fc_before_GRU(outputs)
        return F.relu(final_outputs)


class RGBDMapSimpleCNN(RGBDSimpleCNN):
    def __init__(self, observation_space, config):
        super().__init__(observation_space, config)
        self.embedding_dim = 128
        if 'ego_occ_map' in observation_space.spaces:
            self._ego_occ_map = True
            self._occ_is_color = len(
                observation_space.spaces['ego_occ_map'].shape) > 2
            if self._occ_is_color:
                self.ego_occ_model = SimpleCNN(
                    observation_space, ["ego_occ_map"], 512)
            else:
                bert_config = BertConfig(vocab_size_or_config_json_file=observation_space.spaces["ego_occ_map"].high.max() + 1,
                                         hidden_size=self.embedding_dim,
                                         output_size=self._hidden_size,
                                         max_position_embeddings=observation_space.spaces[
                    "ego_occ_map"].shape[0] // 2,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    intermediate_size=64)
                self.ego_occ_model = BertModel(bert_config)
            self._in_features = self._hidden_size
        else:
            self._ego_occ_map = False

        if 'ego_sem_map' in observation_space.spaces:
            self._ego_sem_map = True
            self._sem_is_color = len(
                observation_space.spaces['ego_sem_map'].shape) > 2
            if self._sem_is_color:
                self.ego_sem_model = SimpleCNN(
                    observation_space, ["ego_sem_map"], 512)
            else:
                bert_config = BertConfig(vocab_size_or_config_json_file=observation_space.spaces["ego_sem_map"].high.max() + 1,
                                         hidden_size=self.embedding_dim,
                                         output_size=self._hidden_size,
                                         max_position_embeddings=observation_space.spaces[
                    "ego_sem_map"].shape[0] // 2,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    intermediate_size=64)
                self.ego_sem_model = BertModel(bert_config)
            self._in_features = self._hidden_size
        else:
            self._ego_sem_map = False

        # if 'scene_dgl' in observation_space.spaces:
        #     self._scene_graph_net = True
        #     self.graph_model = SceneGraphModel(
        #         observation_space, self._hidden_size)
        #     self._in_features += self._hidden_size
        # elif 'scene_graph' in observation_space.spaces:
        #     self._scene_graph_net = True
        #     self.graph_model = SceneGTNModel(
        #         observation_space, self._hidden_size)
        #     self._in_features += self._hidden_size
        # else:
        #     self._scene_graph_net = False
        self._scene_graph_net = False
        # self.fc_before_GRU = nn.Linear(self._in_features, self._hidden_size).cuda(1)
        self.fc_before_GRU = nn.Linear(self._in_features, self._hidden_size)

    def process_map_information(self, map_observations):
        map_observations = map_observations / 255.0
        processed_map_observations = map_observations
        processed_map_observations = processed_map_observations.permute(
            0, 3, 1, 2)
        return processed_map_observations

    def forward(self, observations):
        # outputs = [self.visual_forward(observations)]
        outputs=[]
        if self._ego_occ_map:
            ego_occ_input = observations["ego_occ_map"]
            if self._occ_is_color:
                ego_occ_input = self.process_map_information(ego_occ_input)
            outputs_ego_occ = self.ego_occ_model(ego_occ_input)
            if isinstance(outputs_ego_occ, tuple):
                outputs_ego_occ = outputs_ego_occ[1]
            outputs.append(outputs_ego_occ)

        if self._ego_sem_map:
            ego_sem_input = observations["ego_sem_map"]
            if self._sem_is_color:
                ego_sem_input = self.process_map_information(ego_sem_input)
            outputs_ego_sem = self.ego_sem_model(ego_sem_input)
            
            if isinstance(outputs_ego_sem, tuple):
                # observations['attentions'] = outputs_ego_sem[-1]
                outputs_ego_sem = outputs_ego_sem[1]
            outputs.append(outputs_ego_sem)

        if self._scene_graph_net:
            graph_output = self.graph_model(observations)
            outputs.append(graph_output)

        outputs = torch.cat(outputs, dim=1)
        final_outputs = self.fc_before_GRU(outputs)
        return F.relu(final_outputs)
