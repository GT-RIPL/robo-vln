# Hierarchical Cross-Modal Agent for Vision-and-Language Navigation
<img src="demo/Pytorch_logo.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository is the pytorch implementation of our paper:

**Hierarchical Cross-Modal Agent for Vision-and-Language Navigation**<br>
[__***Muhammad Zubair Irshad***__](https://zubairirshad.com), [Chih-Yao Ma](https://chihyaoma.github.io/), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/) <br>
International Conference on Robotics and Automation (ICRA), 2021<br>

[[Project Page](https://zubair-irshad.github.io/projects/robo-vln.html)] [[arXiv](https://arxiv.org/abs/1901.03035)] [[GitHub](https://github.com/chihyaoma/selfmonitoring-agent)] 

<p align="center">
<img src="demo/ACMI_final.svg" height="260px">
</p>

## Installation - Dependencies

### Habitat and Other Dependencies

SASRA makes extensive use of the Habitat Simulator and API developed by FAIR. You will first need to install both [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-API](https://github.com/facebookresearch/habitat-api/tree/v0.1.4). If you are using conda, Habitat-Sim can easily be installed with:
```bash
conda install -c aihabitat -c conda-forge habitat-sim headless
```
Otherwise, follow the Habitat-Sim [installation instructions](https://github.com/facebookresearch/habitat-sim#installation). Then install Habitat-API version `0.1.4`:

```bash
git clone --branch v0.1.4 git@github.com:facebookresearch/habitat-api.git
cd habitat-api
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```
We recommend downloading the test scenes and running the example script as described [here](https://github.com/facebookresearch/habitat-api/blob/v0.1.4/README.md#installation) to ensure the installation of Habitat-Sim and Habitat-API was successful. Now you can clone this repository and install the rest of the dependencies:
```bash
git clone git@github.com:jacobkrantz/VLN-CE.git
cd VLN-CE
python -m pip install -r requirements.txt
```

### Data

Like Habitat-API, we expect a `data` folder (or symlink) with a particular structure in the top-level directory of this project.

#### Matterport3D

We train and evaluate our agents on Matterport3D (MP3D) scene reconstructions. The official Matterport3D download script (`download_mp.py`) can be accessed by following the "Dataset Download" instructions on their [project webpage](https://niessner.github.io/Matterport/). The scene data needed to run VLN-CE can then be downloaded this way:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Extract this data to `data/scene_datasets/mp3d` such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 total scenes.

#### Dataset

The Robo-VLN dataset is a continuous control formualtion of the VLN-CE dataset by [Krantz et al](https://arxiv.org/pdf/2004.02857.pdf) ported over from Room-to-Room (R2R) dataset created by [Anderson et al](http://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf). The details regarding converting discrete dataset into continuous control formulation can be found in our [paper](https://github.com/zubair-irshad/zubair-irshad.github.io/blob/master/projects/resources/HCM_ICRA21.pdf). 

| Dataset 	| Download path               	| Size  	|
|--------------	|----------------------------	|-------	|
| [robo_vln_v1.zip](https://www.dropbox.com/s/1h1rfx4bssz5qwy/robo_vln_v1.zip?dl=0) 	| `data/datasets/robo_vln_v1`          	| 76.9 MB 	|

#### Robo-VLN Dataset

The dataset `robo_vln_v1` contains the `train`, `val_seen`, and `val_unseen` splits. 

* train: 7739 episodes
* val_seen: 570 episodes
* val_unseen: 1224 episodes

* Format of `{split}.json.gz`

```
{
    'episodes' = [
        {
            'episode_id': 4991,
            'trajectory_id': 3279,
            'scene_id': 'mp3d/JeFG25nYj2p/JeFG25nYj2p.glb',
            'instruction': {
                'instruction_text': 'Walk past the striped area rug...',
                'instruction_tokens': [2384, 1589, 2202, 2118, 133, 1856, 9]
            },
            'start_position': [10.257800102233887, 0.09358400106430054, -2.379739999771118],
            'start_rotation': [0, 0.3332950713608026, 0, 0.9428225683587541],
            'goals': [
                {
                    'position': [3.360340118408203, 0.09358400106430054, 3.07817006111145], 
                    'radius': 3.0
                }
            ],
            'reference_path': [
                [10.257800102233887, 0.09358400106430054, -2.379739999771118], 
                [9.434900283813477, 0.09358400106430054, -1.3061100244522095]
                ...
                [3.360340118408203, 0.09358400106430054, 3.07817006111145],
            ],
            'info': {'geodesic_distance': 9.65537166595459},
        },
        ...
    ],
    'instruction_vocab': [
        'word_list': [..., 'orchids', 'order', 'orient', ...],
        'word2idx_dict': {
            ...,
            'orchids': 1505,
            'order': 1506,
            'orient': 1507,
            ...
        },
        'itos': [..., 'orchids', 'order', 'orient', ...],
        'stoi': {
            ...,
            'orchids': 1505,
            'order': 1506,
            'orient': 1507,
            ...
        },
        'num_vocab': 2504,
        'UNK_INDEX': 1,
        'PAD_INDEX': 0,
    ]
}
```
* Format of `{split}_gt.json.gz`

```
{
    '4991': {
        'actions': [
          ...
          [-0.999969482421875, 1.0],
          [-0.9999847412109375, 0.15731772780418396],
          ...
          ],
        'forward_steps': 325,
        'locations': [
            [10.257800102233887, 0.09358400106430054, -2.379739999771118],
            [10.257800102233887, 0.09358400106430054, -2.379739999771118],
            ...
            [-12.644463539123535, 0.1518409252166748, 4.2241311073303220]
        ]
    }
    ...
}
```

#### Depth Encoder Weights
Similar to [VLN-CE](https://arxiv.org/pdf/2004.02857.pdf), our learning-based models utilizes a depth encoder pretained on a large-scale point-goal navigation task i.e. [DDPPO](https://arxiv.org/abs/1911.00357). We utilize depth pretraining by using the DDPPO features from the ResNet50 from the original paper. The pretrained network can be downloaded [here](https://drive.google.com/open?id=1ueXuIqP2HZ0oxhpDytpc3hpciXSd8H16). Extract the contents of `ddppo-models.zip` to `data/ddppo-models/{model}.pth`.
```bash
# ddppo-models.zip (672M)
gdown https://drive.google.com/uc?id=1ueXuIqP2HZ0oxhpDytpc3hpciXSd8H16
```

## Usage
The `run.py` script is how training and evaluation is done for all model configurations. Specify a configuration file and a run type (either `train` or `eval`) as such:
```bash
python run.py --exp-config path/to/experiment_config.yaml --run-type {train | eval}
```

For example, a random agent can be evaluated on 10 val-seen episodes using this command:
```bash
python run.py --exp-config vlnce_baselines/config/nonlearning.yaml --run-type eval
```

For lists of modifiable configuration options, see the default [task config](habitat_extensions/config/default.py) and [experiment config](vlnce_baselines/config/default.py) files.

### Imitation Learning
For both teacher forcing and DAgger training, experience is collected in simulation and saved to disc for future network updates. This includes saving (at each time step along a trajectory) RGB and Depth encodings, ground truth actions, and instruction tokens. The `DAGGER` config entry allows for specifying which training type is used. A teacher forcing example:

```yaml
DAGGER:
  LR: 2.5e-4  # learning rate
  ITERATIONS: 1  # set to 1 for teacher forcing
  EPOCHS: 15
  UPDATE_SIZE: 10819  # total number of training episodes
  BATCH_SIZE: 5  # number of complete episodes in a batch
  P: 1.0  # Must be 1.0 for teacher forcing
  USE_IW: True  # Inflection weighting
```

A DAgger example:

```yaml
DAGGER:
  LR: 2.5e-4  # learning rate
  ITERATIONS: 15  # number of dataset aggregation rounds
  EPOCHS: 4  # number of network update rounds per iteration
  UPDATE_SIZE: 5000  # total number of training episodes
  BATCH_SIZE: 5  # number of complete episodes in a batch
  P: 0.75  # DAgger: 0.0 < P < 1.0
  USE_IW: True  # Inflection weighting
```

Configuration options exist for loading an already-trained checkpoint for fine-tuning (`LOAD_FROM_CKPT`, `CKPT_TO_LOAD`) as well as for reusing a database of collected features (`PRELOAD_LMDB_FEATURES`, `LMDB_FEATURES_DIR`). Note that reusing collected features for training only makes sense for regular teacher forcing training.

### Evaluating Models
Evaluation of models can be done by running `python run.py --exp-config path/to/experiment_config.yaml --run-type eval`. The relevant config entries for evaluation are:
```bash
EVAL_CKPT_PATH_DIR  # path to a checkpoint or a directory of checkpoints

EVAL.USE_CKPT_CONFIG  # if True, use the config saved in the checkpoint file
EVAL.SPLIT  # which dataset split to evaluate on (typically val_seen or val_unseen)
EVAL.EPISODE_COUNT  # how many episodes to evaluate
```
If `EVAL.EPISODE_COUNT` is equal to or greater than the number of episodes in the evaluation dataset, all episodes will be evaluated. If `EVAL_CKPT_PATH_DIR` is a directory, one checkpoint will be evaluated at a time. If there are no more checkpoints to evaluate, the script will poll the directory every few seconds looking for a new one. Each config file listed in the next section is capable of both training and evaluating the model it is accompanied by.

### Cuda
Cuda will be used by default if it is available. If you have multiple GPUs, you can specify which card is used:
```yaml
SIMULATOR_GPU_ID: 0
TORCH_GPU_ID: 0
NUM_PROCESSES: 1
```
Note that the simulator and torch code do not need to run on the same card. For faster training and evaluation, we recommend running with as many processes (parallel simulations) as will fit on a standard GPU.

## Models and Results From the Paper

| Model              | val_seen SPL | val_unseen SPL | Config                                                                                                                                                                                   |
|--------------------|--------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Seq2Seq            | 0.24         | 0.18           | [seq2seq.yaml](vlnce_baselines/config/paper_configs/seq2seq.yaml)                                                                                                                        |
| Seq2Seq_PM         | 0.21         | 0.15           | [seq2seq_pm.yaml](vlnce_baselines/config/paper_configs/seq2seq_pm.yaml)                                                                                                                  |
| Seq2Seq_DA         | 0.32         | 0.23           | [seq2seq_da.yaml](vlnce_baselines/config/paper_configs/seq2seq_da.yaml)                                                                                                                  |
| Seq2Seq_Aug        | 0.25         | 0.17           | [seq2seq_aug.yaml](vlnce_baselines/config/paper_configs/seq2seq_aug.yaml)  ⟶ [seq2seq_aug_tune.yaml](vlnce_baselines/config/paper_configs/seq2seq_aug_tune.yaml)                         |
| Seq2Seq_PM_DA_Aug  | 0.31         | 0.22           | [seq2seq_pm_aug.yaml](vlnce_baselines/config/paper_configs/seq2seq_pm_aug.yaml)  ⟶ [seq2seq_pm_da_aug_tune.yaml](vlnce_baselines/config/paper_configs/seq2seq_pm_da_aug_tune.yaml) |
| CMA                | 0.25         | 0.22           | [cma.yaml](vlnce_baselines/config/paper_configs/cma.yaml)                                                                                                                                |
| CMA_PM             | 0.26         | 0.19           | [cma_pm.yaml](vlnce_baselines/config/paper_configs/cma_pm.yaml)                                                                                                                          |
| CMA_DA             | 0.31         | 0.25           | [cma_da.yaml](vlnce_baselines/config/paper_configs/cma_da.yaml)                                                                                                                          |
| CMA_Aug            | 0.24         | 0.19           | [cma_aug.yaml](vlnce_baselines/config/paper_configs/cma_aug.yaml)  ⟶ [cma_aug_tune.yaml](vlnce_baselines/config/paper_configs/cma_aug_tune.yaml)                                         |
| **CMA_PM_DA_Aug**  | **0.35**     | **0.30**       | [cma_pm_aug.yaml](vlnce_baselines/config/paper_configs/cma_pm_aug.yaml)  ⟶ [cma_pm_da_aug_tune.yaml](vlnce_baselines/config/paper_configs/cma_pm_da_aug_tune.yaml)                 |
| CMA_PM_Aug         | 0.25         | 0.22           | [cma_pm_aug.yaml](vlnce_baselines/config/paper_configs/cma_pm_aug.yaml)  ⟶ [cma_pm_aug_tune.yaml](vlnce_baselines/config/paper_configs/cma_pm_aug_tune.yaml)                             |
| CMA_DA_Aug         | 0.33         | 0.26           | [cma_aug.yaml](vlnce_baselines/config/paper_configs/cma_aug.yaml)  ⟶ [cma_da_aug_tune.yaml](vlnce_baselines/config/paper_configs/cma_da_aug_tune.yaml)                             |


|         |  Legend                                                                                                                                               |
|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Seq2Seq | Sequence-to-Sequence baseline model                                                                                                                   |
| CMA     | Cross-Modal Attention model                                                                                                                           |
| PM      | [Progress monitor](https://github.com/chihyaoma/selfmonitoring-agent)                                                                                 |
| DA      | DAgger training (otherwise teacher forcing)                                                                                                           |
| Aug     | Uses the [EnvDrop](https://github.com/airsplay/R2R-EnvDrop) episodes to augment the training set                                                      |
| ⟶       | Use the config on the left to train the model. Evaluate each checkpoint on `val_unseen`. The best checkpoint (according to `val_unseen` SPL) is then fine-tuned using the config on the right. Make sure to update the field `DAGGER.CKPT_TO_LOAD` before fine-tuning. |

### Pretrained Models
We provide pretrained models for our best Seq2Seq model [Seq2Seq_DA](https://drive.google.com/open?id=1gds-t8LAxuh236gk-5AWU0LzDg9rJmQS) and Cross-Modal Attention model ([CMA_PM_DA_Aug](https://drive.google.com/open?id=199hhL9M0yiurB3Hb_-DrpMRxWP1lSGX3)). These models are hosted on Google Drive and can be downloaded as such:
```bash
python -m pip install gdown

# CMA_PM_DA_Aug (141MB)
gdown https://drive.google.com/uc?id=199hhL9M0yiurB3Hb_-DrpMRxWP1lSGX3
# Seq2Seq_DA (135MB)
gdown https://drive.google.com/uc?id=1gds-t8LAxuh236gk-5AWU0LzDg9rJmQS
```

## Contributing
This codebase is under the MIT license. If you find something wrong or have a question, feel free to open an issue. If you would like to contribute, please install pre-commit before making commits in a pull request:
```bash
python -m pip install pre-commit
pre-commit install
```

## Citing
