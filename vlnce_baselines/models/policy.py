from habitat_baselines.common.utils import CustomFixedCategorical
from habitat_baselines.rl.ppo.policy import Policy
from habitat_extensions.ppo import TransformerPolicy 


class BasePolicy(Policy):
    '''
    Base Policy for Seq2Seq Models. Inherits habitat-api RL PPO Policy
    '''
    def build_distribution(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        return distribution

class BaseTransformerPolicy(TransformerPolicy):
    def build_distribution(
        self, observations, prev_actions_list, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, prev_actions_list = self.net(
            observations, prev_actions_list, prev_actions, masks
        )
        self.action_distribution = self.action_distribution.cuda(0)
        distribution = self.action_distribution(features)

        return distribution

