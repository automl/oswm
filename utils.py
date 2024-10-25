import os
import math
import argparse
import random
import datetime
import itertools
from typing import Union

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


def torch_to_python_float(x):
    return float(x.detach().cpu().numpy())


def get_cosine_schedule_with_warmup(warmup_share, max_lr):
    def lr_lambda(curr_share):
        if curr_share < warmup_share:
            return max_lr * (curr_share / warmup_share)
        else:
            decay_share = (curr_share - warmup_share) / (1.0 - warmup_share)
            return (
                max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * decay_share)))
                * max_lr
            )

    return lr_lambda


def set_lr(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


def get_openai_lr(transformer_model):
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)


def get_weighted_single_eval_pos_sampler(max_len, min_len=0, p=1.0):
    """
    This gives a sampler that can be used for `single_eval_pos` which yields good performance for all positions p,
    where p <= `max_len`. At most `max_len` - 1 examples are shown to the Transformer.
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(
        range(min_len, max_len),
        [1 / math.pow(((max_len - min_len) - i), p) for i in range(max_len - min_len)],
    )[0]


def get_uniform_single_eval_pos_sampler(max_len, min_len=0):
    """
    Just sample any evaluation position with the same weight
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(min_len, max_len))[0]


class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


def set_locals_in_self(locals):
    """
    Call this function like `set_locals_in_self(locals())` to set all local variables as object variables.
    Especially useful right at the beginning of `__init__`.
    :param locals: `locals()`
    """
    self = locals["self"]
    for var_name, val in locals.items():
        if var_name != "self":
            setattr(self, var_name, val)


default_device = "cuda:0" if torch.cuda.is_available() else "cpu:0"


def is_cuda(device_or_device_str: Union[torch.device, str]):
    return torch.device(device_or_device_str).type == "cuda"


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    else:
        return obj


# Copied from StackOverflow, but we do an eval on the values additionally
class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            try:
                my_dict[k] = eval(v)
            except NameError:
                my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
        print("dict values: {}".format(my_dict))


def get_nan_value(v, set_value_to_nan=1.0):
    if random.random() < set_value_to_nan:
        return v
    else:
        return random.choice([-999, 0, 1, 999])


def to_ranking(data):
    x = data >= data.unsqueeze(-3)
    x = x.sum(0)
    return x


# TODO: Is there a better way to do this?
#   1. Cmparing to unique elements: When all values are different we still get quadratic blowup
#   2. Argsort(Argsort()) returns ranking, but with duplicate values there is an ordering which is problematic
#   3. Argsort(Argsort(Unique))->Scatter seems a bit complicated, doesn't have quadratic blowup, but how fast?
def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = data[:, :, col] >= data[:, :, col].unsqueeze(-2)
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def nan_handling_missing_for_unknown_reason_value(nan_prob=1.0):
    return get_nan_value(float("nan"), nan_prob)


def nan_handling_missing_for_no_reason_value(nan_prob=1.0):
    return get_nan_value(float("-inf"), nan_prob)


def nan_handling_missing_for_a_reason_value(nan_prob=1.0):
    return get_nan_value(float("inf"), nan_prob)


def torch_nanmean(x, axis=0, return_nanshare=False):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    if return_nanshare:
        return value / num, 1.0 - num / x.shape[axis]
    return value / num


def torch_nanstd(x, axis=0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis), x.shape[axis], dim=axis
    )
    return torch.sqrt(
        torch.nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1)
    )


def normalize_data(
    data, normalize_positions=-1, return_scaling=False, clip=True, std_only=False
):
    """
    Normalize data to mean 0 and std 1

    :param data: T,B,H
    :param normalize_positions: If > 0, only use the first `normalize_positions` positions for normalization
    :param return_scaling: If True, return the scaling parameters as well (mean, std)
    :param clip: If True, clip values to [-100, 100]
    :param std_only: If True, only divide by std
    """
    if normalize_positions is not None and normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], axis=0)
        std = torch_nanstd(data[:normalize_positions], axis=0) + 0.000001
    else:
        mean = torch_nanmean(data, axis=0)
        std = torch_nanstd(data, axis=0) + 0.000001

    if std_only:
        mean = 0
    data = (data - mean) / std

    if clip:
        data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)
    return data


def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"
    # for b in range(X.shape[1]):
    # for col in range(X.shape[2]):
    data = X if normalize_positions == -1 else X[:normalize_positions]
    data_clean = data[:].clone()
    data_mean, data_std = torch_nanmean(data, axis=0), torch_nanstd(data, axis=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
    data_mean, data_std = torch_nanmean(data_clean, axis=0), torch_nanstd(
        data_clean, axis=0
    )
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X


def bool_mask_to_att_mask(mask):
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def print_on_master_only(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(device):
    print("init dist")
    if "LOCAL_RANK" in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print("torch.distributed.launch and my rank is", rank)
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
            "only I can print, but when using print(..., force=True) it will print on all ranks."
        )
        return True, rank, f"cuda:{rank}"
    elif "SLURM_PROCID" in os.environ and torch.cuda.device_count() > 1:
        # this is for multi gpu when starting with submitit
        assert device != "cpu:0"
        rank = int(os.environ["SLURM_PROCID"])
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        print("distributed submitit launch and my rank is", rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
            "only I can print, but when using print(..., force=True) it will print on all ranks."
        )

        return True, rank, f"cuda:{rank}"
    else:
        print("Not using distributed")
        # will not change any of the behavior of print, but allows putting the force=True in the print calls
        print_on_master_only(True)
        return False, 0, device


# NOP decorator for python with statements (x = NOP(); with x:)
class NOP:
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def check_compatibility(dl):
    if hasattr(dl, "num_outputs"):
        print(
            "`num_outputs` for the DataLoader is deprecated. It is assumed to be 1 from now on."
        )
        assert dl.num_outputs != 1, (
            "We assume num_outputs to be 1. Instead of the num_ouputs change your loss."
            "We specify the number of classes in the CE loss."
        )


def product_dict(dic):
    keys = dic.keys()
    vals = dic.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def to_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return torch.tensor(x, device=device)


printed_already = set()


def print_once(*msgs: str):
    msg = " ".join([repr(m) for m in msgs])
    if msg not in printed_already:
        print(msg)
        printed_already.add(msg)


def check_file_exists(path):
    """Checks if a pickle file exists. Returns None if not, else returns the unpickled file."""
    if os.path.isfile(path):
        # print(f'loading results from {path}')
        with open(path, "rb") as f:
            try:
                return np.load(f, allow_pickle=True).tolist()
            except:
                return None
    return None


def rank_values(x):
    s = torch.argsort(x, descending=False, dim=0)
    x = torch.arange(x.numel(), device=x.device)[torch.argsort(s, dim=0)]
    return x


def map_unique_to_order(x):
    # Alternative implementation:
    # boolean = output[:, None] == torch.unique(output)
    # output = torch.nonzero(boolean)[:, -1]
    if len(x.shape) != 2:
        raise ValueError("map_unique_to_order only works with 2D tensors")
    if x.shape[1] > 1:
        return torch.cat(
            [map_unique_to_order(x[:, i : i + 1]) for i in range(x.shape[1])], dim=1
        )
    print(x.shape)
    return (x > torch.unique(x).unsqueeze(0)).sum(1).unsqueeze(-1)


"""timings with GPU involved are potentially wrong.
TODO: a bit of documentation on how to use these.
maybe include torch.cuda.synchronize!? but might make things slower..
maybe better write that timings with GPU involved are potentially wrong.
"""
import time

timing_dict_aggregation, timing_dict, timing_meta_dict = {}, {}, {}


def timing_start(name="", enabled=True, meta=None):
    if not enabled:
        return
    timing_dict[name] = time.time()
    timing_meta_dict[name] = meta


def timing_end(name="", enabled=True, collect=False):
    if not enabled:
        return
    if collect:
        timing_dict_aggregation[name] = (
            timing_dict_aggregation.get(name, 0) + time.time() - timing_dict[name]
        )
    else:
        print("Timing", name, time.time() - timing_dict[name], timing_meta_dict[name])
        timing_meta_dict[name] = None


def lambda_time(f, name="", enabled=True, collect=False, meta=None):
    timing_start(name, enabled, meta=meta)
    r = f()
    timing_end(name, enabled, collect)
    return r


def lambda_time_flush(name="", enabled=True):
    if not enabled or name not in timing_dict_aggregation:
        return
    print("Timing", name, timing_dict_aggregation[name])
    timing_dict_aggregation[name] = 0
