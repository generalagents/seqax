from dataclasses import dataclass
import enum
import zarr
import numpy as np
import concurrent
from numcodecs import Blosc, Delta


class Split(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass
class Config:
    tokens_chunk_size: int
    seq_starts_chunk_size: int
    _target_: str = __name__ + ".Config"


from dataclasses import dataclass


@dataclass
class Chunk:
    """An in-memory encoding of flat tokens. Use this as a buffer for writing to a FlatTokensWriter."""

    image_values: (
        np.ndarray
    )  # uint32[num_tokens] - this array keeps track of the image patches. Start and end of the image patches are determined by the patch size.
    encoded_tokens: (
        np.ndarray
    )  # uint32[num_tokens] - this array keeps track of the text in the document
    document_tokens: (
        np.ndarray
    )  # uint32[num_tokens] - this array keeps track of which tokens are images and which are text in the document
    seq_starts: (
        np.ndarray
    )  # uint64[num_seqs] - this array keeps track of the start of each document
    max_text_token_id: int
    patch_size: tuple[
        int, int
    ]  # size of the image patches - this helps to decide the start and end of the image patches
    max_image_token_id: int  # the maximum token id for the image tokens

    @staticmethod
    def from_ragged(
        image_values: list[np.ndarray],
        text_tokens: list[np.ndarray],
        doc_tokens: list[np.ndarray],
        max_text_token_id: int,
        patch_size: tuple[int, int],
        max_image_token_id: int,
    ):
        """
        Convert a list of sequences (document tokens) to a FlatTokensChunk.
        
        - image_values: will be all the values = these can't be of different lengths. they will depend on patch size. [num_images, patch_size[0] * patch_size[1]]
        - doc_tokens: can be of varying lengths. [num_docs, num_tokens]
        
        # TODO: consider if text tokens is a redundancy.
        """
        document_tokens = np.concatenate(doc_tokens)
        seq_starts = np.zeros(len(doc_tokens) + 1, np.uint64)
        np.cumsum([len(seq) for seq in doc_tokens], out=seq_starts[1:])
        in_bounds_seq_starts = np.where(
            seq_starts != len(document_tokens), seq_starts, 0
        )

        # document tokens will need to be multiplied by 2.
        document_tokens <<= 1
        document_tokens[
            in_bounds_seq_starts
        ] |= 1  # this ensures that the start of the sequence is marked.

        return Chunk(
            image_values=np.vstack(image_values).flatten(),
            encoded_tokens=np.concatenate(text_tokens),
            document_tokens=document_tokens,
            seq_starts=seq_starts,
            max_text_token_id=max_text_token_id,
            patch_size=patch_size,
            max_image_token_id=max_image_token_id,
        )


class Writer:
    def __init__(self, filespec: str, split: Split, mode: str, config: Config):
        try:
            dst_root = zarr.open_group(filespec, mode=mode, cache_attrs=True)
        except zarr.errors.ContainsGroupError:
            raise ValueError(f"Output {filespec} already exists.")
        self.group = dst_root.require_group(split.value)

        # Use BITSHUFFLE for encoded_tokens, since the token IDs will typically only be ~14-17 bits wide.
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

        if "max_token_id" not in self.group.attrs:
            self.group.attrs["max_token_id"] = 0

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

    def write(self, chunk: Chunk):
        """Synchronously writes a chunk of flat tokens to the underlying storage.

        Results are committed when this function returns, no need for a separate close() or
        flush() call. This function should not be called concurrently for the same destination.

        You typically want to call this in a separate thread, to overlap computation with I/O.
        """
        num_tokens = self.encoded_tokens.shape[0]
        if chunk.max_token_id > self.group.attrs["max_token_id"]:
            self.group.attrs["max_token_id"] = chunk.max_token_id
        # In parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(lambda: self.encoded_tokens.append(chunk.encoded_tokens))
            executor.submit(
                lambda: self.seq_starts.append(num_tokens + chunk.seq_starts[1:])
            )
