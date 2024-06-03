"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from argparse import Namespace
from typing import MutableMapping

import torch
import logging

from torch import Tensor


def default_batch_size():
    if torch.cuda.is_available():
        batch_size = int(
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1000 * 0.3
        )
        if batch_size == 0:
            logging.warning("GPU VRAM is too small. Computing on CPU.")
    elif torch.backends.mps.is_available():
        # I don't know if there's an equivalent API so heuristically choosing bs=4
        batch_size = 4
    else:
        # don't know what a good value is here. Would not recommend to run on CPU
        batch_size = 1
        logging.warning("No GPU found. Conversion on CPU is very slow.")
    return batch_size


def move_to_device(model, bf16: bool = True, cuda: bool = True):
    try:
        if torch.backends.mps.is_available():
            return model.to("mps")
    except AttributeError:
        pass
    if bf16:
        model = model.to(torch.bfloat16)
    if cuda and torch.cuda.is_available():
        model = model.to("cuda")
    return model


def tensor_sizes(input=None, **kwargs) -> ...:
    """
    A very useful method to inspect the sizes of tensors in object containing Tensors
    Args:
        input ():
        **kwargs ():

    Returns:

    """
    if kwargs:
        return tensor_sizes(kwargs)

    if isinstance(input, (dict, MutableMapping)):
        return {key: tensor_sizes(v) \
                for key, v in input.items()}
    if isinstance(input, Namespace):
        return {key: tensor_sizes(v) \
                for key, v in input.__dict__.items()}

    elif isinstance(input, tuple):
        return tuple(tensor_sizes(v) for v in input)
    elif isinstance(input, list):
        if len(input) and isinstance(input[0], str):
            return len(input)
        return [tensor_sizes(v) for v in input]
    elif isinstance(input, set):
        if len(input) and isinstance(list(input)[0], str):
            return len(input)
        elif len(input) and isinstance(list(input)[0], tuple):
            return ['.'.join(tup) for tup in input]
        return len(input)

    else:
        if input is not None and hasattr(input, "shape"):
            if isinstance(input, Tensor) and input.dim() == 0:
                return input.item()

            return list(input.shape)
        else:
            return input
