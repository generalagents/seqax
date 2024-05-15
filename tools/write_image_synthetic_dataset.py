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
    max_patches_per_image: int
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
) -> Float[np.ndarray, "..."]:
    """Generates a random image of a given size.

    Returns:
        np.ndarray: A random image of the given size.
    """
    img_array = gen.integers(0, 256, size=size)
    return img_array.astype(np.float32)


@jaxtyped(typechecker=typechecker)
def generate_patches(
    image: np.ndarray, patch_size: tuple[int, int]
) -> Float[np.ndarray, "..."]:
    """
    Takes an image and returns patches of the image.
    """
    patches = view_as_blocks(image, block_shape=patch_size)
    patches = patches.reshape(
        -1, patch_size[0], patch_size[1]
    )  # (num_patches, patch_size[0], patch_size[1])
    return patches.astype(np.float32)


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
    max_patches_per_image = config.max_patches_per_image

    # init arrays for storing intermediates
    all_text_tokens = []
    all_document_tokens = []

    # current max_image_token_id
    max_image_token_id = max_text_token_id
    patches_array = np.zeros(
        (
            num_docs,
            num_images_per_doc,
            max_patches_per_image,
            patch_size[0],
            patch_size[1],
        )
    )
    patches_array.fill(-1)  # fill with -1 to indicate no image

    for doc_id in range(num_docs):
        # generate the text tokens
        text_tokens = generate_text_tokens(
            num_text_tokens_per_doc, max_text_token_id, gen
        )
        all_text_tokens.append(text_tokens)

        # generate the images and their patches
        num_images_random = np.random.randint(
            1, num_images_per_doc + 1
        )  # generate this random number of images for this document
        for img_id in range(num_images_per_doc):
            image = generate_image(image_size, gen)
            patches = generate_patches(image, patch_size)
            num_patches_to_store = min(max_patches_per_image, patches.shape[0])

            # Store the patches in the preallocated array
            patches_array[doc_id, img_id, :num_patches_to_store] = patches[
                :num_patches_to_store
            ]

        # interleave the text and image tokens
        image_tokens = np.arange(
            max_image_token_id,
            max_image_token_id + num_images_random,
            dtype=np.uint32,
        )
        max_image_token_id = max_image_token_id + num_images_random

        # interleave to create document tokens
        document_tokens = interleave_tokens(text_tokens, image_tokens, gen)
        all_document_tokens.append(document_tokens)

    return image_flat_tokens.Chunk.from_ragged(
        patches_array,
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
        dst.write(examples_chunk)


if __name__ == "__main__":
    main()
