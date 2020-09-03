import torch
import torch.nn as nn
from habitat_baselines.common.utils import CategoricalNet
import pytorch_lightning as pl


class TransformerPolicyLightning(pl.LightningModule):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        prev_actions_list,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, prev_actions_list = self.net(
            observations, prev_actions_list, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, prev_actions_list

    def get_value(self, observations, prev_actions_list, prev_actions, masks):
        features, _ = self.net(
            observations, prev_actions_list, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, prev_actions_list, prev_actions, masks, action
    ):
        features, prev_actions_list = self.net(
            observations, prev_actions_list, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, prev_actions_list
class TransformerPolicy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        prev_actions_list,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, prev_actions_list = self.net(
            observations, prev_actions_list, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, prev_actions_list

    def get_value(self, observations, prev_actions_list, prev_actions, masks):
        features, _ = self.net(
            observations, prev_actions_list, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, prev_actions_list, prev_actions, masks, action
    ):
        features, prev_actions_list = self.net(
            observations, prev_actions_list, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, prev_actions_list


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)