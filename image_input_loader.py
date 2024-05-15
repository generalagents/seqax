"""Input data loading from `flat-tokens` data format.

See `docs/flat-tokens.md` for details on the format.

We support shuffling of the input data, by the following algorithm:
* there are N independent "streams" of data, each of which has disjoint data and is
  shuffled independently.
* within each stream, we fetch a "shuffle buffer" consisting of many "read blocks" of
  data. We shuffle the entire buffer in memory.
* the "read blocks" attached to each shuffle buffer are themselves selected randomly.

This is the standard shuffling used by e.g. Huggingface Datasets. Unlike them, we run
this algorithm _after_ tokenization, so we know exactly at which step number each new
shuffle buffer starts at, allowing us to do instant resumes after job restarts. In our
default recommended configuration, we also recommend a much larger shuffle buffer size
than Huggingface Datasets, which allows for more thorough shuffling, taking advantage
of the fact that a single sequence of tokens uses very little memory compared to e.g.
a single image.

Mosaic's StreamingDatasets library uses a similar algorithm as us, which they call py1b: 
https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/shuffling.html.
"""

from concurrent.futures import ThreadPoolExecutor
import functools
from typing import Tuple, Union, Optional

from typeguard import typechecked
from shardlib.shardtypes import bool_, pytree_dataclass, u32, f32
import shardlib.shardtypes as shardtypes
import zarr
from dataclasses import dataclass
import jax
import numpy as np
from jax.sharding import PartitionSpec as P
import datetime
import jax

# imports for hf dataloader
import numpy as onp
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset


@dataclass(frozen=True)
class TokenBatchParams:
    """The shape of a token batch."""

    len: int
    batch: int


import numpy as np
from dataclasses import dataclass


@pytree_dataclass
class TokenBatch:
    """
    A batch of tokens, which are typically the input to training.
        tokens: u32['batch/d max_token_length']
    patches: f32['batch/d patch_size patch_size num_channels']
    indices: u32['batch/d seqlen']

    indices = 0 1 2 3 4 5 6 7 8 9
    tokens[0] patches[0] tokens[1] patches[1]
    [tokens[i // 2] if i % 2 == 0 else patches[i // 2] for i in indices] - in the model
    """

    tokens: u32["batch/d len"]  # uint32['batch/d max_token_length']
    patches: f32[
        "batch/d patchsize patchsize numchannels"
    ]  # float32['batch/d patch_size patch_size num_channels']
    indices: u32["batch/d len"]  # uint32['batch/d seqlen']
