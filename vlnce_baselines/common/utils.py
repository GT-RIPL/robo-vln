from typing import Dict, List
from transformers import BertTokenizer
import torch
from transformers import AutoTokenizer

from tokenizers import (ByteLevelBPETokenizer,
                            CharBPETokenizer,
                            SentencePieceBPETokenizer,
                            BertWordPieceTokenizer)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"


def get_bert_tokens(sentence, max_seq_length, tokenizer):
    output = tokenizer.encode(sentence)
    # tokens = tokenizer.tokenize(sentence)
    # tokens = ['[CLS]'] + tokens + ['[SEP]']
    # if len(tokens) < max_seq_length:
    #     padded_tokens = tokens + ['[PAD]' for _ in range(max_seq_length - len(tokens))]
    # token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    return output.ids

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
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if is_bert:
        instruction = observations[
            instruction_sensor_uuid
        ]["text"]
        token_ids = get_bert_tokens(instruction , max_seq_length, tokenizer)
        observations[instruction_sensor_uuid] = token_ids
        # observations[i][instruction_sensor_uuid] = observations[i][
        #     instruction_sensor_uuid
        # ]["tokens"]
    else:
        observations[instruction_sensor_uuid] = observations[
            instruction_sensor_uuid
        ]["tokens"]

    return observations

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

def get_semantic_mask(input_ids):
    attention_mask = input_ids > 0
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = ~extended_attention_mask
    return extended_attention_mask

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

