import os
from pathlib import Path
import platform
import random
from glob import glob

import torchvision.transforms as transforms

from image_generation.shaders.on_the_fly_moderngl_shader import ModernGLOnlineDataset
from torch.utils.data.dataloader import DataLoader


def get_shader_dir_path(twigl: bool):
    """
    Only used if shader's parent dir isn't given
    twigl - whether to use twigl or shadertoy
    """
    if platform.system() == "Windows":
        prefix = "C:\\Users\\owner\\Desktop\\shader_codes\\shader_codes"
    else:
        prefix = "/content/gdrive/My Drive/shaders/shader_codes"

    if twigl:
        return os.path.join(prefix, "twigl")
    else:
        return os.path.join(prefix, "shadertoy")


def split_data(parent_dir=None, seed=0, ratios=(0.7, 0.1, 0.2)):
    """
    parent_dir: shadertoy or twigl directory. if None, chooses twigl and will attempt to resolve path by itself
    split data with a seed.
    NOTE: the seed argument isn't used in any func call in the code. The seed from main args doesn't effect split.
    """
    if parent_dir is None:
        parent_dir = get_shader_dir_path(twigl=True)
    elif parent_dir in ["twigl", "shadertoy"]:
        parent_dir = get_shader_dir_path(twigl=(parent_dir == "twigl"))

    assert os.path.isdir(parent_dir), f"parent dir does not exists {parent_dir}. (assuming {platform.system()})"
    assert len(ratios) == 3 and sum(ratios) == 1.0, f"data split ratios for train/val/test are illegal {ratios}"

    if Path(parent_dir).stem == "shadertoy":
        all_shaders = glob(os.path.join(parent_dir, "*", "*.fragment"))
        # shadertoy's file hierarchy is "shadertoy/random_letter/some_id.fragment
    elif Path(parent_dir).stem == "twigl":
        all_shaders = glob(os.path.join(parent_dir, "codes", "*.fragment"))
        # twigl's file hierarchy is "twigl/either 'codes' or 'modes'/some_id.fragment
    else:
        raise ValueError(f"parent dir is neither shadertoy nor twigl, {parent_dir}")

    # randomize data fold by seed
    all_shaders = sorted(all_shaders, key=os.path.basename)
    random.Random(seed).shuffle(all_shaders)

    n = len(all_shaders)
    train_shaders = all_shaders[:int(n*ratios[0])]
    val_shaders = all_shaders[int(n*ratios[0]):int(n*(ratios[0] + ratios[1]))]
    test_shaders = all_shaders[int(n*(ratios[0] + ratios[1])):]

    print(f"{n} shaders found. splits:")
    print(f"{len(train_shaders)} train. {train_shaders[:5]}...")
    print(f"{len(val_shaders)} val. {val_shaders[:5]}...")
    print(f"{len(test_shaders)} test. {test_shaders[:5]}...")

    return train_shaders, val_shaders, test_shaders


def get_dataloader(twigl: bool, is_train: bool, n: int, gpu: int, batch_size: int, limit_shaders: int, resolution: int):
    """
    get dataloader for training a (C)VAE model.
    limit_shaders - due to RAM memory limit, using all 21k shaders can cause crashes/slowdown. instead each set will
                    only use up to this many shaders
    """
    train, val, _ = split_data("twigl" if twigl else "shadertoy")
    shaders = train if is_train else val

    if limit_shaders:
        shaders = shaders[:limit_shaders]

    dataset = ModernGLOnlineDataset(shaders,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize([0.5, 0.5, 0.5],
                                                                                       [0.5, 0.5, 0.5])]),
                                    resolution=resolution,
                                    max_queue_size=batch_size*10,
                                    n_samples=-1 if is_train else n,
                                    parallel=True,
                                    gpus=[gpu],
                                    virtual_dataset_size=n,
                                    generate_extra_repeats=is_train)
    # when n_samples is -1, get_item ignores the index and samples a new image at random
    # when n_samples>=1, the dataset samples randomly for the first n_samples images and saves them for future epochs

    # we don't need to shuffle=True in training because the training dataset generates data online
    # we don't need increased num_workers because the parallelization occurs in the dataset
    return DataLoader(dataset, batch_size=batch_size, drop_last=is_train)


def get_dataloader_test(twigl: bool, gpu: int, batch_size: int, limit_shaders: int, resolution: int):
    """
    same as get_dataloader, see documentation there, except virtual dataset size==num shaders, and train==False.
    """
    _, _, test = split_data("twigl" if twigl else "shadertoy")
    shaders = test

    if limit_shaders:
        shaders = shaders[:limit_shaders]

    dataset = ModernGLOnlineDataset(shaders,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize([0.5, 0.5, 0.5],
                                                                                       [0.5, 0.5, 0.5])]),
                                    resolution=resolution,
                                    max_queue_size=batch_size*10,
                                    n_samples=len(shaders),
                                    parallel=True,
                                    gpus=[gpu],
                                    virtual_dataset_size=len(shaders),
                                    generate_extra_repeats=False)
    # when n_samples is -1, get_item ignores the index and samples a new image at random
    # when n_samples>=1, the dataset samples randomly for the first n_samples images and saves them for future epochs
    return DataLoader(dataset, batch_size=batch_size)
