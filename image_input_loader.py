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
    Attributes:
        tokens (np.array): Array of tokens for the batch.
            Shape: (batch_size, sequence_length)

        patches (np.array): Array of flattened image patches for the batch.
            Shape: (num_patches, patch_h * patch_w)

        indices (np.array): Array indicating whether each token is an image (1) or text (0).
            Shape: (batch_size, sequence_length)
    """

    tokens: u32["batch/d len"]  # uint32['batch/d max_token_length']
    patches: f32[
        "batch/d patch_h*patch_w"
    ]  # float32['batch/d patch_size*patch_size*num_channels']
    indices: u32["batch/d len"]  # uint32['batch/d max_token_length']


class ZarrImageTextLoader:

    def __init__(
        self,
        split: str,
        params: FlatTokensParams,
        token_batch_params: TokenBatchParams,
        num_channels: int = 3,
    ):
        """
        Initializes the ImageInputLoader object.

        Args:
            split (str): The split of the dataset to load. Must be either "train" or "validation".
            params (FlatTokensParams): Parameters for loading the dataset.
            token_batch_params (TokenBatchParams): Parameters for batching tokens.
            num_channels (int, optional): The number of channels in the image. Defaults to 3.
        """
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
        """
        Load a batch of tokens, patches, and indices for a given step.

        Args:
            step (int): The step number.

        Returns:
            TokenBatch: An instance of the TokenBatch class containing the loaded tokens, patches, and indices.
        """
        start_idx = step * self.token_batch_params.batch
        end_idx = start_idx + self.token_batch_params.batch

        if end_idx > self.seq_count:
            raise IndexError(
                f"Requested step {step} exceeds available data with batch size {self.token_batch_params.batch}."
            )

        batch_tokens = []
        batch_indices = []
        batch_patches = []

        patch_counter = 0
        new_line_token = -1

        for doc_idx in range(start_idx, end_idx):
            doc_start = self.seq_starts[doc_idx]
            # print(f"doc_start: {doc_start}")
            doc_end = (
                self.seq_starts[doc_idx + 1]
                if doc_idx + 1 < len(self.seq_starts)
                else self.document_tokens.shape[0]
            )
            # print(f"doc_end: {doc_end}")

            doc_tokens = self.document_tokens[doc_start:doc_end]
            doc_images = self.patch_values[doc_idx]
            # print("doc tokens:", doc_tokens)

            tokens = []
            indices = []
            local_image_id = 0
            for token in doc_tokens:
                if token >= self.max_text_token_id:
                    # print("\nIMAGE TOKEN")
                    image_patches = doc_images[local_image_id]
                    # print(f"image_patches: {image_patches.shape}")
                    local_image_id += 1
                    rasterized_patches, raster_tokens, raster_indices = (
                        self.rasterize_patches_fast(
                            image_patches, patch_counter, new_line_token
                        )
                    )
                    # print("Rasterized patches: ", rasterized_patches.shape)
                    # print(rasterized_patches)
                    # print("===" * 30)
                    # print(raster_tokens)
                    # print("\n\n\n")
                    batch_patches.extend(rasterized_patches)
                    tokens.extend(raster_tokens)
                    indices.extend(raster_indices)
                    patch_counter += len(rasterized_patches)
                else:
                    indices.append(0)
                    tokens.append(token)

            batch_tokens.append(tokens)
            batch_indices.append(indices)

        return TokenBatch(
            tokens=batch_tokens, patches=batch_patches, indices=batch_indices
        )

    def rasterize_patches(self, image_patches, patch_counter, new_line_token):
        max_num_patches, patch_h, patch_w = image_patches.shape
        rasterized_patches = []
        raster_tokens = []
        raster_indices = []
        for i in range(0, max_num_patches, patch_w):
            for j in range(patch_w):
                if i + j < max_num_patches:
                    patch = image_patches[i + j]
                    rasterized_patches.append(patch.flatten())
                    raster_tokens.append(patch_counter)
                    raster_indices.append(1)
                    patch_counter += 1
            raster_tokens.append(new_line_token)
            raster_indices.append(0)
        return (
            np.array(rasterized_patches),
            np.array(raster_tokens),
            np.array(raster_indices),
        )

    def rasterize_patches_fast(self, image_patches, patch_counter, new_line_token):
        """
        Rasterizes the given image patches into a 2D array and returns the rasterized patches, raster tokens, and raster indices.

        Args:
            image_patches (ndarray): The input image patches to be rasterized. The shape of the array should be (max_num_patches, patch_h, patch_w).
            patch_counter (int): The starting value for the raster tokens.
            new_line_token (int): The token value to represent a new line in the rasterized patches.

        Returns:
            tuple: A tuple containing the rasterized patches, raster tokens, and raster indices.
                - rasterized_patches (ndarray): The rasterized image patches. The shape of the array is (num_patches, patch_h * patch_w).
                - raster_tokens (ndarray): The tokens representing the rasterized patches. The shape of the array is (num_patches + num_newlines,).
                - raster_indices (ndarray): The indices indicating whether a token represents a patch or a new line. The shape of the array is (num_patches + num_newlines,).

        """
        max_num_patches, patch_h, patch_w = image_patches.shape
        rasterized_patches = image_patches.reshape(-1, patch_h * patch_w)
        num_patches = rasterized_patches.shape[0]

        raster_tokens = np.arange(patch_counter, patch_counter + num_patches)
        raster_indices = np.ones(num_patches, dtype=int)

        positions = np.arange(patch_w, num_patches, patch_w)
        newline_positions = positions + np.arange(len(positions))

        raster_tokens = np.insert(raster_tokens, newline_positions, new_line_token)
        raster_indices = np.insert(raster_indices, newline_positions, 0)

        if raster_tokens[-1] != new_line_token:
            raster_tokens = np.append(raster_tokens, new_line_token)
            raster_indices = np.append(raster_indices, 0)

        return rasterized_patches, raster_tokens, raster_indices


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
