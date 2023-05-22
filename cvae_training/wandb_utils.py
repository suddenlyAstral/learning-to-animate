import os
from glob import glob
from shutil import rmtree

import numpy as np
import torch


def path_to_wandb_dir():
    if os.path.isdir("wandb"):
        return "wandb"
    else:
        return os.path.join("cvae_training", "wandb")


def remove_old_wandb_runs():
    old_runs_regex = os.path.join(path_to_wandb_dir(), "run-*")
    old_run_dirs = glob(old_runs_regex)
    for directory in old_run_dirs:
        rmtree(directory)


def remove_old_wandb_media():
    old_runs_regex = os.path.join(path_to_wandb_dir(), "run-*", "files", "media")
    all_media_dirs = glob(old_runs_regex)
    # if we've deleted old runs, this should only be one dir, but just in case we delete all
    for directory in all_media_dirs:
        rmtree(directory)


def tensor2np(t: torch.Tensor, permute=True, from_zero_one=False):
    """
    changes a tensor image/image batch to an np array in [0,255]
    permute: True if we need to move C from (C, H, W) to be (H, W, C)
    from_zero_one: True if the tensor is in [0,1]. otherwise, assumed in [-1,1]
    """
    t = t.detach().cpu()
    if permute:
        if t.dim() == 4:
            t = t.permute(0, 2, 3, 1)
        else:
            t = t.permute(1, 2, 0)
    if not from_zero_one:
        # we assume the tensor is normalized for network, i.e. in [-1,1], rather than [0,1]
        # we can't just normalize so highest is 255 and lowest 0 because images aren't guaranteed to include either
        t = (0.5*t) + 0.5
    t = (t.numpy()*255).round().astype(np.uint8)
    return t
