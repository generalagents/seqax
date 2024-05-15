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


@dataclass(frozen=True)
class FlatTokensParams:
    filespec: str
    seed: int
    sequence_packing: bool


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


class ZarrImageTextLoader:

    def __init__(
        self,
        split: str,
        params: FlatTokensParams,
        token_batch_params: TokenBatchParams,
        num_channels: int = 3,
    ):
        self.params = params
        self.token_batch_params = token_batch_params
        self.root = zarr.open_group(params.filespec, mode="r")
        assert split in ["train", "validation"], "Invalid split"
        split = f"Split.{split.capitalize()}"
        self.encoded_tokens = self.root[split]["encoded_tokens"]
        self.seq_starts = self.root[split]["seq_starts"]
        self.patch_values = self.root[split]["patch_values"]
        self.document_tokens = self.root[split]["document_tokens"]
        self.max_text_token_id = self.root[split].attrs["max_text_token_id"]
        self.patch_size = tuple(self.root[split].attrs["patch_size"])
        print("Patch size: ", self.patch_size)
        print("Patch Values: ", self.patch_values.shape)
        print("Document Tokens: ", self.document_tokens[:])
        self.num_channels = num_channels

        assert len(self.encoded_tokens.shape) == 1, "Expected 1D zarr"
        assert self.encoded_tokens.dtype == np.uint32, "Expected uint32 zarr"
        assert len(self.seq_starts.shape) == 1, "Expected 1D zarr"
        assert self.seq_starts.dtype == np.uint64, "Expected uint64 zarr"

        self.seq_count = self.seq_starts.shape[0] - 1
        print("Number of sequences: ", self.seq_count)

    def load(self, step: int) -> TokenBatch:
        # Calculate the start and end indices for the batch
        start_idx = step * self.token_batch_params.batch
        end_idx = start_idx + self.token_batch_params.batch

        print("Indices requested: ", start_idx, end_idx)

        if end_idx > self.seq_count:
            raise IndexError(
                f"Requested step {step} exceeds available data with batch size {self.token_batch_params.batch}."
            )

        # Get the sequence start indices for the batch
        seq_start_indices = self.seq_starts[start_idx:end_idx]
        print("Sequence start indices: ", seq_start_indices)

        # Prepare the tokens, patches, and indices arrays

        batch_tokens = []
        batch_patches = []
        batch_indices = []
        batch_image_id = 0
        for doc_idx in range(start_idx, end_idx):
            doc_start = self.seq_starts[doc_idx]
            doc_end = (
                self.seq_starts[doc_idx + 1]
                if doc_idx + 1 < len(self.seq_starts)
                else self.document_tokens.shape[0]
            )
            print(doc_idx, doc_start, doc_end)

            doc_tokens = self.document_tokens[doc_start:doc_end]
            print(doc_tokens)

            doc_images = self.patch_values[
                doc_idx
            ]  # [max_num_images_per_doc, num_patches, patch_size, patch_size]

            tokens = []
            indices = []
            local_image_id = 0
            for token in doc_tokens:
                if token >= self.max_text_token_id:
                    # This is an image - indices will be 1
                    indices.append(1)
                    # get the image
                    image = doc_images[local_image_id]
                    local_image_id += 1

                    # add image to the batch images
                    batch_patches.append(image)  # [num_patches, patch_size, patch_size]
                    batch_image_id += 1

                    # add the image index to the tokens
                    tokens.append(batch_image_id)

                else:
                    # This is a text token - indices will be 0
                    indices.append(0)
                    # add token as it is
                    tokens.append(token)

            batch_tokens.append(tokens)
            batch_indices.append(indices)

        return TokenBatch(
            tokens=batch_tokens, patches=batch_patches, indices=batch_indices
        )


if __name__ == "__main__":
    # Load the dataset
    params = FlatTokensParams(
        filespec="synthetic_image_dataset.zarr",
        seed=0,
        sequence_packing=False,
    )
    token_batch_params = TokenBatchParams(len=1, batch=1)
    loader = ZarrImageTextLoader("train", params, token_batch_params)

    # Load a batch
    batch = loader.load(0)
    print("Tokens: ", batch.tokens)
    print("Indices: ", batch.indices)
    print("Patches: ", batch.patches)
