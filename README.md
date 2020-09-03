# SASRA: Semantically-aware Spatio-temporal Reasoning Agent for Vision and Language Navigation
<img src="demo/Pytorch_logo.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository is the pytorch implementation of end to end interpretable trasnformer like architecture for Vision and Language Navigation:
<img align="right" src="demo/einstein_scroll.png" width="5%">
<img align="right" src="demo/SRI_Logo.png" width="6%">

<p align="center">
  <img width="800" height="450" src="demo/hybrid_arch.png" alt="Hybrid Transformer-RNN Model for VLN">
</p>


### Running different configurations:

#### Some general Comments:

- All configurations run on a single GPU except Hybrid CMA which utlises Model Parallel ( To run model parallel, add the following line of code in `SASRA/vlnce_baselines/models/policy.py` line 27 in `BaseTransformerPolicy`:
`self.action_distribution = self.action_distribution.cuda(0)`. Uncomment this line if you are not running Model Parallel ( True for all configurations except hybrid_cma.yaml)
)

- Make sure to only set `PRELOAD_LMDB_FEATURES=FALSE` when creating a new dataset for any configurations. If you are using existing datasets available under `VLNCE-data/data/trajectory_dir` (The path is already set for existing datasets in the specific configurations), make sure to set `PRELOAD_LMDB_FEATURES=TRUE`. If you do not explicitly set this to `TRUE`, the already collected data would be corrupted

- While running configurations which support `dagger_trainer.py` (Please see below all configurations running on dagger_trainer.py), you do not need to change model name inside dagger trainer. If you are running configurations running on `transformer_trainer.py`, you need to explictly set which model you want to use. An example is detailed here: 

1. Import the Model you want to use:

from vlnce_baselines.models.transformer_semantics_policy import TransformerSemPolicy

2. Explictly set the Model in Line 301 in transformer_trainer.py as follows: 

            self.actor_critic = TransformerSemPolicyMP(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                model_config=config,
                batch_size=self.config.DAGGER.BATCH_SIZE,
                gpus = [0,1,2]
            )

3. For `dagger trainer.py`, double check to see if you using the correct policy by looking at lines 284-316  


- Running Eval Scripts: For conigurations using `transformer_trainer.py`, please make sure to set `NUM_PROCESSES = 1` for correct evaluation. For all conigurations using `dagger_trainer.py` except `hybrid_cma.yaml`, you can set `NUM_PROCESSES > 1`. For `hybrid_cma.yaml`, since the training script uses Model Parallel, create a new hybrid_cma_eval polocy by copying `hybrid_cma_mp.py` in another file and disabling all model parallel gpu shifts. Use this policy in `dagger_trainer.py` while evaluating `hybrid_cma` results with `NUM_PROCESSES = 1`

## Recently experimented Models (These are in addition to the ones available in the VLN-CE codebase)


| Model                 | Trainer used           | Model File                      | Model Name              | Config                                                                                        |
|-----------------------|------------------------|---------------------------------|-------------------------|-----------------------------------------------------------------------------------------------|
| Seq2Seq               | dagger_trainer.py      | seq2seq_policy.py               | Seq2SeqPolicy           | [seq2seq.yaml](vlnce_baselines/config/paper_configs/seq2seq.yaml)                             |
| Seq2Seq_SEM_ATTN      | dagger_trainer.py      | seq2seq_sem_attn.py             | Seq2Seq_Sem_Attn_Polic  | [seq2seq_pm.yaml](vlnce_baselines/config/paper_configs/seq2seq_sem_attn.yaml)                 |
| TRANSFORMER           | dagger_trainer.py      | transformer_policy.py           | TransformerPolicy       | [seq2seq_text_attn.yaml](vlnce_baselines/config/paper_configs/seq2seq_text_attn.yaml)         |
| TRANSFORMER_SEMANTICS | transformer_trainer.py | transformer_semantics_policy.py | TransformerSemPolicy    | [transformer_semantics.yaml](vlnce_baselines/config/paper_configs/transformer_semantics.yaml) |
| HYBRID_CMA            | transformer_trainer.py | hybrid_cma_mp.py                | TransformerHybridPolicy | [hybrid_cma.yaml](vlnce_baselines/config/paper_configs/hybrid_cma.yaml)                       |

|                                                                                                        


## Setup

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
The R2R_VLNCE dataset is a port of the Room-to-Room (R2R) dataset created by [Anderson et al](http://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf) for use with the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) (MP3D-Sim). For details on the porting process from MP3D-Sim to the continuous reconstructions used in Habitat, please see our [paper](https://arxiv.org/abs/2004.02857). We provide two versions of the dataset, `R2R_VLNCE_v1` and `R2R_VLNCE_v1_preprocessed`. `R2R_VLNCE_v1` contains the `train`, `val_seen`, and `val_unseen` splits. `R2R_VLNCE_v1_preprocessed` runs with our models out of the box. It includes instruction tokens mapped to GloVe embeddings, ground truth trajectories, and a data augmentation split (`envdrop`) that is ported from [R2R-EnvDrop](https://github.com/airsplay/R2R-EnvDrop). For more details on the dataset contents and format, see our [project page](https://jacobkrantz.github.io/vlnce/data).

| Dataset 	| Extract path               	| Size  	|
|--------------	|----------------------------	|-------	|
| [R2R_VLNCE_v1.zip](https://drive.google.com/file/d/1k9LLJGeDGLIO2wxtWhzjGHhvQ2t2aJBQ/view) 	| `data/datasets/R2R_VLNCE_v1`          	| 3 MB 	|
| [R2R_VLNCE_v1_preprocessed.zip](https://drive.google.com/file/d/1IDM4eEMTJDKN6-mGTSmqSkv620hCd_TX/view)  	| `data/datasets/R2R_VLNCE_v1_preprocessed` 	| 344 MB 	|

Downloading the dataset:
```bash
python -m pip install gdown
cd data/datasets

# R2R_VLNCE_v1
gdown https://drive.google.com/uc?id=1k9LLJGeDGLIO2wxtWhzjGHhvQ2t2aJBQ
unzip R2R_VLNCE_v1.zip
rm R2R_VLNCE_v1.zip

# R2R_VLNCE_v1_preprocessed
gdown https://drive.google.com/uc?id=1IDM4eEMTJDKN6-mGTSmqSkv620hCd_TX
unzip R2R_VLNCE_v1_preprocessed.zip
rm R2R_VLNCE_v1_preprocessed.zip
```

#### Encoder Weights
The learning-based models receive a depth observation at each time step. The depth encoder we use is a ResNet pretrained on a PointGoal navigation task using [DDPPO](https://arxiv.org/abs/1911.00357). In this work, we extract features from the ResNet50 trained on Gibson 2+ from the original paper, whose weights can be downloaded [here](https://drive.google.com/open?id=1ueXuIqP2HZ0oxhpDytpc3hpciXSd8H16). Extract the contents of `ddppo-models.zip` to `data/ddppo-models/{model}.pth`.
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
