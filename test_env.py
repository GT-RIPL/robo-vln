from vlnce_baselines.common.env_utils import construct_envs,construct_envs_auto_reset_false
from habitat import Config
from vlnce_baselines.config.default import get_config
from habitat_baselines.common.environments import get_env_class
import torch
import time
from habitat_baselines.common.baseline_registry import baseline_registry


def spawn_env(config):
    envs = None

    device = (
    torch.device("cuda", 0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

    actions = torch.zeros(2, 1, device=device)
    if envs==None:
        envs =  construct_envs(config, get_env_class(config.ENV_NAME))
    obs=envs.reset()
    print(obs)
    object2idx = envs.call_at(0,'get_object2idx')
    print(object2idx)
    envs.close()
if __name__ == '__main__':
    exp_config ='vlnce_baselines/config/paper_configs/seq2seq.yaml'
    config = get_config(exp_config, None)

    split = config.TASK_CONFIG.DATASET.SPLIT
    config.defrost()
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split

    # if doing teacher forcing, don't switch the scene until it is complete
    if config.DAGGER.P == 1.0:
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
    config.freeze()
    spawn_env(config)
