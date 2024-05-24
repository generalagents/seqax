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

import functools
from typing import Tuple
import tensorflow as tf
from src.seqax.shardlib.shardtypes import pytree_dataclass, u32, f32, bool_
import src.seqax.shardlib.shardtypes as shardtypes
from dataclasses import dataclass
import jax
from src.data.constants import LATEST_TFRECORD_EXPORT_GS_URI
from src.data.tfrecord import episode_from_tfrecord_generator, tokenize_episodes, make_examples

@dataclass(frozen=True)
class TokenBatchParams:
    """The shape of a token batch."""
    len: int
    batch: int


@pytree_dataclass
class TokenBatch:
    patches: f32['batch/d len h w c']
    tokens: u32['batch/d len']
    indices: u32['batch/d len']
    is_starts: bool_['batch/d len']


class GeneralAgentsDataLoader:
    def __init__(self,):
        self.batch_size = 8
        self.max_seq_len = max_seq_len = 128
        self.shardings = shardtypes.make_shardings(TokenBatch)
        def generator():
            return make_examples(tokenize_episodes(episode_from_tfrecord_generator(LATEST_TFRECORD_EXPORT_GS_URI)), max_seq_len=max_seq_len)
        dataset = tf.data.Dataset.from_generator(
            generator, output_signature=dict(
                patches=tf.TensorSpec(shape=(max_seq_len, 256, 256, 3), dtype=tf.float32, name=None),
                tokens=tf.TensorSpec(shape=(max_seq_len,), dtype=tf.uint32, name=None),
                indices=tf.TensorSpec(shape=(max_seq_len,), dtype=tf.uint32, name=None),
                is_starts=tf.TensorSpec(shape=(max_seq_len,), dtype=tf.bool, name=None),
                )
            )
        dataset = dataset.shuffle(16).batch(self.batch_size, drop_remainder=True)
        self.iterator = dataset.as_numpy_iterator()

    def load(self, step):
        batch = next(self.iterator)
        def get_shard(x: jax.Array, indexing: Tuple[slice]) -> jax.Array:
            shard = x[indexing]
            return shard
        patches = jax.make_array_from_callback((self.batch_size, self.max_seq_len, 256, 256, 3), self.shardings.patches, functools.partial(get_shard, batch['patches']))
        tokens = jax.make_array_from_callback((self.batch_size, self.max_seq_len), self.shardings.tokens, functools.partial(get_shard, batch['tokens']))
        indices = jax.make_array_from_callback((self.batch_size, self.max_seq_len), self.shardings.indices, functools.partial(get_shard, batch['indices']))
        is_starts = jax.make_array_from_callback((self.batch_size, self.max_seq_len), self.shardings.is_starts, functools.partial(get_shard, batch['is_starts']))
        return TokenBatch(patches, tokens, indices, is_starts)


def get_loader():
    return GeneralAgentsDataLoader()