"""Input data loading from `image-flat-tokens` data format.
"""

from dataclasses import dataclass

import numpy as np
import zarr
from shardlib.shardtypes import bool_, f32, pytree_dataclass, u32


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
        self.num_channels = num_channels

        assert len(self.encoded_tokens.shape) == 1, "Expected 1D zarr"
        assert self.encoded_tokens.dtype == np.uint32, "Expected uint32 zarr"
        assert len(self.seq_starts.shape) == 1, "Expected 1D zarr"
        assert self.seq_starts.dtype == np.uint64, "Expected uint64 zarr"

        self.seq_count = self.seq_starts.shape[0] - 1

    def load(self, step: int) -> TokenBatch:
        # Calculate the start and end indices for the batch
        start_idx = step * self.token_batch_params.batch
        end_idx = start_idx + self.token_batch_params.batch

        if end_idx > self.seq_count:
            raise IndexError(
                f"Requested step {step} exceeds available data with batch size {self.token_batch_params.batch}."
            )

        # Get the sequence start indices for the batch
        seq_start_indices = self.seq_starts[start_idx : end_idx + 1]

        # Preallocate the tokens, patches, and indices arrays
        total_docs = end_idx - start_idx
        max_seq_len = np.diff(seq_start_indices).max()
        batch_tokens = np.zeros((total_docs, max_seq_len), dtype=np.uint32)
        batch_indices = np.zeros((total_docs, max_seq_len), dtype=np.uint32)
        batch_patches = []

        # Fill the arrays with data
        for doc_idx in range(total_docs):
            doc_start = seq_start_indices[doc_idx]
            doc_end = seq_start_indices[doc_idx + 1]

            doc_tokens = self.document_tokens[doc_start:doc_end]

            # Process tokens and patches
            image_local_idx = 0
            for token_idx, token in enumerate(doc_tokens):
                if token >= self.max_text_token_id:
                    batch_indices[doc_idx, token_idx] = 1  # indicates image
                    batch_patches.append(
                        self.patch_values[start_idx + doc_idx, image_local_idx]
                    )
                    batch_tokens[doc_idx, token_idx] = (
                        len(batch_patches) - 1
                    )  # index of the patch in batch_patches
                    image_local_idx += 1
                else:
                    # indicates text (no change to indices as init as 0)
                    batch_tokens[doc_idx, token_idx] = token

        # Convert patches to a numpy array
        batch_patches = np.array(batch_patches, dtype=np.float32)

        return TokenBatch(
            tokens=batch_tokens, patches=batch_patches, indices=batch_indices
        )


def get_loader(
    split: str,
    config: FlatTokensParams,
    token_batch_params: TokenBatchParams,
):
    if isinstance(config, FlatTokensParams):
        return ZarrImageTextLoader(split, config, token_batch_params)
    else:
        raise ValueError(f"Unknown config type {type(config)}")


if __name__ == "__main__":
    # Load the dataset
    params = FlatTokensParams(
        filespec="tools/synthetic_image_dataset.zarr",
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
