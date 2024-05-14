# To run:
#
# ```
# python write_image_synthetic_dataset.py --config-name=image_synthetic_dataset +output=synthetic_image_dataset.zarr
# ```
#
# Synthetic tasks: see Section 4 of https://arxiv.org/abs/2002.09402 for some ideas.
#
# We do:
# * Task 3: [ours] text with images.
#
# Sequences begin with task ID, then have task-specific data. We avoid index 0, which indicates padding.

from functools import partial
import hydra
from jaxtyping import Float, Int, jaxtyped, UInt32, Array
import numpy as np
from typeguard import typechecked as typechecker
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import image_flat_tokens
from skimage.util import view_as_blocks


@dataclass
class Config:
    output: str
    seed: int
    seq_len: int
    examples: int
    max_text_token_id: int
    image_size: int
    patch_size: int
    images_per_example: int
    flat_tokens_config: image_flat_tokens.Config
    _target_: str = __name__ + ".Config"


@jaxtyped(typechecker=typechecker)
def generate_text_tokens(
    seq_len: int, max_text_token_id: int, gen: np.random.Generator
) -> UInt32[np.ndarray, "seqlen"]:
    seq = gen.integers(1, max_text_token_id, ((seq_len + 1) // 2), dtype=np.uint32)
    return np.append(seq, np.flip(seq), axis=0)[:seq_len]


@jaxtyped(typechecker=typechecker)
def generate_image(
    size: tuple[int, int], gen: np.random.Generator
) -> UInt32[np.ndarray, "..."]:
    """Generates a random image of a given size.

    Returns:
        np.ndarray: A random image of the given size.
    """
    img_array = gen.integers(0, 256, size=size, dtype=np.uint32)
    return img_array


@jaxtyped(typechecker=typechecker)
def generate_patches(
    image: np.ndarray, patch_size: tuple[int, int]
) -> UInt32[np.ndarray, "..."]:
    """
    Takes an image and returns patches of the image.
    """
    patches = view_as_blocks(image, block_shape=patch_size)
    patches = patches.reshape(-1, patch_size[0] * patch_size[1])
    return patches


@jaxtyped(typechecker=typechecker)
def interleave_tokens(
    text_tokens: np.ndarray, image_tokens: np.ndarray, gen: np.random.Generator
) -> np.ndarray:
    # randomly select positions for the image tokens
    num_text_tokens = len(text_tokens)
    num_image_tokens = len(image_tokens)

    combined_length = num_text_tokens + num_image_tokens
    document_tokens = np.zeros(combined_length, dtype=np.uint32)

    # randomly select indices for image tokens
    image_indices = gen.choice(combined_length, num_image_tokens, replace=False)

    text_index = 0
    for i in range(combined_length):
        if i in image_indices:
            document_tokens[i] = image_tokens[np.where(image_indices == i)[0][0]]
        else:
            if text_index < num_text_tokens:
                document_tokens[i] = text_tokens[text_index]
                text_index += 1
    return document_tokens


@jaxtyped(typechecker=typechecker)
def synthetic_task(config: Config, gen: np.random.Generator) -> image_flat_tokens.Chunk:
    num_text_tokens_per_doc = config.seq_len - 1
    num_docs = config.examples
    num_images_per_doc = config.images_per_example
    image_size = (config.image_size, config.image_size)
    patch_size = (config.patch_size, config.patch_size)
    max_text_token_id = config.max_text_token_id

    # init arrays for storing intermediates
    all_image_patches = []
    all_text_tokens = []
    all_document_tokens = []

    # current max_image_token_id
    max_image_token_id = max_text_token_id

    for docid in range(num_docs):
        # generate the text tokens
        text_tokens = generate_text_tokens(
            num_text_tokens_per_doc, max_text_token_id, gen
        )
        all_text_tokens.append(text_tokens)

        # generate the images and their patches
        all_image_patches_per_doc = []
        for _ in range(num_images_per_doc):
            image = generate_image(image_size, gen)
            patches = generate_patches(image, patch_size)
            all_image_patches_per_doc.append(patches.flatten())
        all_image_patches_per_doc = np.vstack(
            all_image_patches_per_doc
        )  # (num_images_per_doc , patch_size[0] * patch_size[1])
        all_image_patches.append(all_image_patches_per_doc)

        # interleave the text and image tokens
        image_tokens = np.arange(
            max_image_token_id,
            max_image_token_id + len(all_image_patches_per_doc),
            dtype=np.uint32,
        )
        max_image_token_id = max_image_token_id + len(all_image_patches_per_doc)

        # interleave to create document tokens
        document_tokens = interleave_tokens(text_tokens, image_tokens, gen)
        all_document_tokens.append(document_tokens)

    return image_flat_tokens.Chunk.from_ragged(
        all_image_patches,
        all_text_tokens,
        all_document_tokens,
        max_text_token_id,
        patch_size,
        max_image_token_id,
    )


# Registering the Config class with the name 'config'.
ConfigStore.instance().store(name="config_schema", node=Config)


@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = hydra.utils.instantiate(config)
    gen = np.random.Generator(np.random.PCG64(config.seed))

    for split, mode in [
        (image_flat_tokens.Split.VALIDATION, "w-"),
        (image_flat_tokens.Split.TRAIN, "r+"),
    ]:
        dst = image_flat_tokens.Writer(
            config.output, split, mode, config.flat_tokens_config
        )
        examples_chunk = synthetic_task(config, gen)
        print(examples_chunk)
        # save to numpy
        # np.savez(config.output, examples_chunk)
        dst.write(examples_chunk)


if __name__ == "__main__":
    main()
