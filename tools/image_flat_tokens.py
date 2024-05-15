import concurrent
import enum
from dataclasses import dataclass
import tensorflow as tf

import numpy as np
import zarr
from numcodecs import Blosc, Delta
import tf_utils


class Split(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass
class Config:
    tokens_chunk_size: int
    seq_starts_chunk_size: int
    image_chunk_size: int
    _target_: str = __name__ + ".Config"


@dataclass
class Chunk:
    """An in-memory encoding of flat tokens. Use this as a buffer for writing to a FlatTokensWriter."""

    patch_values: (
        np.ndarray
    )  # float[num_docs num_images_per_doc num_patches patch_size patch_size] - this array keeps track of the image patches. -1 indicates no image.
    encoded_tokens: (
        np.ndarray
    )  # uint32[num_docs * num_text_tokens_per_doc] - this array keeps track of the text in the document. TODO: probably redundant, consider removing.
    document_tokens: (
        np.ndarray
    )  # uint32[num_docs * (num_text_tokens_per_doc + num_image_tokens_per_doc)] - this array keeps track of which tokens are images and which are text in the document
    seq_starts: (
        np.ndarray
    )  # uint64[num_seqs] - this array keeps track of the start of each document
    max_text_token_id: int
    patch_size: tuple[int, int]  # size of the image patches
    max_image_token_id: int  # the maximum token id for the image tokens

    @staticmethod
    def from_ragged(
        patch_values: list[np.ndarray],
        text_tokens: list[np.ndarray],
        doc_tokens: list[np.ndarray],
        max_text_token_id: int,
        patch_size: tuple[int, int],
        max_image_token_id: int,
    ):
        """
        Convert a list of sequences (document tokens) to a FlatTokensChunk.

        - patch_values: will be all the values = these can't be of different lengths. they will depend on patch size. [num_images, patch_size[0] * patch_size[1]]
        - doc_tokens: can be of varying lengths. [num_docs, num_tokens]

        # TODO: consider if text tokens is a redundancy.
        """
        document_tokens = np.concatenate(doc_tokens)
        seq_starts = np.zeros(len(doc_tokens) + 1, np.uint64)
        np.cumsum([len(seq) for seq in doc_tokens], out=seq_starts[1:])
        in_bounds_seq_starts = np.where(
            seq_starts != len(document_tokens), seq_starts, 0
        )

        return Chunk(
            patch_values=np.array(patch_values, dtype=np.float32),
            encoded_tokens=np.concatenate(text_tokens),
            document_tokens=document_tokens,
            seq_starts=in_bounds_seq_starts,
            max_text_token_id=max_text_token_id,
            patch_size=patch_size,
            max_image_token_id=max_image_token_id,
        )


class Writer:
    def __init__(self, filespec: str, split: str, mode: str, config):
        try:
            dst_root = zarr.open_group(filespec, mode=mode, cache_attrs=True)
        except zarr.errors.ContainsGroupError:
            raise ValueError(f"Output {filespec} already exists.")
        self.group = dst_root.require_group(split)

        # Use BITSHUFFLE for encoded_tokens, since the token IDs will typically only be ~14-17 bits wide.
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

        if "max_text_token_id" not in self.group.attrs:
            self.group.attrs["max_text_token_id"] = 0

        if "max_image_token_id" not in self.group.attrs:
            self.group.attrs["max_image_token_id"] = 0

        if "encoded_tokens" in self.group:
            self.encoded_tokens = self.group["encoded_tokens"]
        else:
            self.encoded_tokens = self.group.empty(
                "encoded_tokens",
                shape=(0,),
                chunks=(config.tokens_chunk_size,),
                dtype=np.uint32,
                compressor=compressor,
            )

        if "seq_starts" in self.group:
            self.seq_starts = self.group["seq_starts"]
        else:
            # Use delta encoding for seq_starts, since they're known to be sorted.
            filters = [Delta(dtype="i8")]
            self.seq_starts = self.group.zeros(
                "seq_starts",
                shape=(1,),
                chunks=(config.seq_starts_chunk_size,),
                dtype=np.uint64,
                compressor=compressor,
                filters=filters,
            )

        if "document_tokens" in self.group:
            self.document_tokens = self.group["document_tokens"]
        else:
            self.document_tokens = self.group.empty(
                "document_tokens",
                shape=(0,),
                chunks=(config.tokens_chunk_size,),
                dtype=np.uint32,
                compressor=compressor,
            )

        self.compressor = compressor

    def write(self, chunk: Chunk):
        """Synchronously writes a chunk of flat tokens to the underlying storage.

        Results are committed when this function returns, no need for a separate close() or
        flush() call. This function should not be called concurrently for the same destination.

        You typically want to call this in a separate thread, to overlap computation with I/O.
        """
        if "patch_values" not in self.group:
            # Create patch_values with the initial chunk's shape
            self.patch_values = self.group.zeros(
                "patch_values",
                shape=chunk.patch_values.shape,
                dtype=np.float32,
                compressor=self.compressor,
            )
            self.patch_values[:] = chunk.patch_values
        else:
            self.patch_values = self.group["patch_values"]
            # Ensure shapes are compatible, resize if necessary
            if self.patch_values.shape != chunk.patch_values.shape:
                # Adjusting only the first dimension to append data
                new_shape = list(self.patch_values.shape)
                new_shape[0] += chunk.patch_values.shape[0]
                self.patch_values.resize(new_shape)
                # Append the new data along the first dimension
            self.patch_values.append(chunk.patch_values)

        num_tokens = self.encoded_tokens.shape[0]
        if chunk.max_text_token_id > self.group.attrs["max_text_token_id"]:
            self.group.attrs["max_text_token_id"] = chunk.max_text_token_id
        if chunk.max_image_token_id > self.group.attrs["max_image_token_id"]:
            self.group.attrs["max_image_token_id"] = chunk.max_image_token_id
        if "patch_size" not in self.group.attrs:
            self.group.attrs["patch_size"] = chunk.patch_size

        # In parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(lambda: self.encoded_tokens.append(chunk.encoded_tokens))
            executor.submit(
                lambda: self.seq_starts.append(num_tokens + chunk.seq_starts[1:])
            )

            executor.submit(lambda: self.document_tokens.append(chunk.document_tokens))


import tf_utils


class TFWriter:

    def __init__(self, filename: str, split: str):
        self.filename = f"{filename}_{split}.tfrecord"
        self.writer = tf.io.TFRecordWriter(self.filename)

    def serialize_example(self, chunk: Chunk):

        feature = {
            "patch_values": tf_utils._bytes_feature(
                tf.io.serialize_tensor(chunk.patch_values).numpy()
            ),
            "encoded_tokens": tf_utils._bytes_feature(
                tf.io.serialize_tensor(chunk.encoded_tokens).numpy()
            ),
            "document_tokens": tf_utils._bytes_feature(
                tf.io.serialize_tensor(chunk.document_tokens).numpy()
            ),
            "seq_starts": tf_utils._bytes_feature(
                tf.io.serialize_tensor(chunk.seq_starts).numpy()
            ),
            "max_text_token_id": tf_utils._int64_feature(chunk.max_text_token_id),
            "patch_size": tf_utils._bytes_feature(
                tf.io.serialize_tensor(chunk.patch_size).numpy()
            ),
            "max_image_token_id": tf_utils._int64_feature(chunk.max_image_token_id),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def write(self, chunk: Chunk):

        self.writer.write(self.serialize_example(chunk))

    def close(self):
        self.writer.close()
