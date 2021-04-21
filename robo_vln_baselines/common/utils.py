from typing import Dict, List
from transformers import BertTokenizer
import torch
from transformers import AutoTokenizer
from collections import defaultdict
from tokenizers import (ByteLevelBPETokenizer,
                            CharBPETokenizer,
                            SentencePieceBPETokenizer,
                            BertWordPieceTokenizer)

import os
import numpy as np

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "True"
def get_bert_tokens(sentence, max_seq_length, tokenizer):
    output = tokenizer.encode(sentence)
    return output.ids

def _to_tensor(v) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

def batch_obs_data_collect(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(_to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = (
            torch.stack(batch[sensor], dim=0)
            .to(device=device)
            .to(dtype=torch.float)
        )

    return batch

def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for sensor in observations:
        batch[sensor].append(_to_tensor(observations[sensor]))

    for sensor in batch:
        batch[sensor] = (
            torch.stack(batch[sensor], dim=0)
            .to(device=device)
            .to(dtype=torch.float)
        )

    return batch

def transform_obs(
    observations: Dict, instruction_sensor_uuid: str, is_bert = False, max_seq_length = 200
) -> Dict[str, torch.Tensor]:
    r"""Extracts instruction tokens from an instruction sensor and
    transposes a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        instruction_sensor_uuid: name of the instructoin sensor to
            extract from.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    tokenizer = BertWordPieceTokenizer("vocab_files/bert-base-uncased-vocab.txt", lowercase=True)
    if is_bert:
        observations['glove_tokens'] = observations[instruction_sensor_uuid]["tokens"]

        instruction = observations[
            instruction_sensor_uuid
        ]["text"]
        token_ids = get_bert_tokens(instruction , max_seq_length, tokenizer)
        observations[instruction_sensor_uuid] = token_ids
    else:
        observations[instruction_sensor_uuid] = observations[
            instruction_sensor_uuid
        ]["tokens"]

    return observations

def split_batch_tbptt(batch, prev_actions, not_done_masks, corrected_actions, oracle_stop_batch, tbptt_steps, split_dim):
    new_observations_batch = defaultdict(list)
    split_observations_batch = defaultdict()
    batch_split=[]
    for sensor in batch:
        if sensor == 'instruction':
            new_observations_batch[sensor] = batch[sensor]
            continue
        for x in batch[sensor].split(tbptt_steps, dim=split_dim):
            new_observations_batch[sensor].append(x)            
    for i, (prev_action_split, corrected_action_split, masks_split, oracle_stop_split) in enumerate(
            zip(prev_actions.split(tbptt_steps, dim=split_dim), 
                  corrected_actions.split(tbptt_steps, dim=split_dim), 
                  not_done_masks.split(tbptt_steps, dim=split_dim),
                  oracle_stop_batch.split(tbptt_steps, dim=split_dim) )):
            for sensor in new_observations_batch:
                if sensor == 'instruction':
                    split_observations_batch[sensor] = new_observations_batch[sensor]
                else:
                    split_observations_batch[sensor] = new_observations_batch[sensor][i]
            split = (split_observations_batch, prev_action_split, masks_split, corrected_action_split, oracle_stop_split)
            split_observations_batch = {}
            batch_split.append(split)
            
    return batch_split

def repackage_mini_batch(batch):
    split_observations_batch, prev_action_split, not_done_masks, corrected_action_split = batch
    for sensor in split_observations_batch:
        split_observations_batch[sensor] = split_observations_batch[sensor].contiguous().view(
            -1, *split_observations_batch[sensor].size()[2:]
        )
    return (
        split_observations_batch,
        prev_action_split.contiguous().view(-1, 2),
        not_done_masks.contiguous().view(-1, 2),
        corrected_action_split.contiguous().view(-1,2),
    )

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

def get_transformer_mask(instr_embedding, instr_len):
    mask = torch.ones((instr_embedding.shape[0], instr_embedding.shape[1]), dtype=torch.bool).cuda()
    attention_mask = torch.ones((instr_embedding.shape[0], instr_embedding.shape[1], instr_embedding.shape[1]), dtype=torch.bool).cuda()
    for i, _len in enumerate(instr_len):
        mask[i, :_len] = 0
        attention_mask[i, :_len, :_len] = 0
    pe_mask = mask.unsqueeze(dim=-1)
    value = (instr_embedding, pe_mask)
    return value, attention_mask.unsqueeze(1), mask.unsqueeze(dim=1).unsqueeze(dim=1)


def get_instruction_mask(instr_embedding, instr_len):
    mask = torch.ones((instr_embedding.shape[0], instr_embedding.shape[1]), dtype=torch.bool).cuda()
    for i, _len in enumerate(instr_len):
        mask[i, :_len] = 0
    return mask.unsqueeze(dim=1).unsqueeze(dim=1)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

