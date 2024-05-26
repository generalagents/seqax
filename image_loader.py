import tensorflow as tf
from google.cloud import storage
import functools
import dataclasses


def decode_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  return image


def pad_image(image, patch_size):
  img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]
  pad_height = (patch_size - img_height % patch_size) % patch_size
  pad_width = (patch_size - img_width % patch_size) % patch_size
  padded_image = tf.image.pad_to_bounding_box(image, 0, 0, img_height + pad_height, img_width + pad_width)
  return padded_image
  

def image_to_patches(image, patch_size):
  patches = tf.image.extract_patches(
    images=tf.expand_dims(image, 0),
    sizes=[1, patch_size, patch_size, 1],
    strides=[1, patch_size, patch_size, 1],
    rates=[1, 1, 1, 1],
    padding='VALID')
  patches = tf.reshape(patches, [-1, patch_size, patch_size, 3])
  return patches


def image_to_patch_sequences(image_path, *, patch_size):
  image = decode_image(image_path)
  padded_image = pad_image(image, patch_size)
  patches = image_to_patches(padded_image, patch_size)
  num_patches = tf.shape(patches)[0]
  is_start = tf.concat([[True], tf.fill([num_patches - 1], False)], axis=0)
  return dict(patches=patches, is_start=is_start)


def gs_glob(bucket_name, prefix):
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blobs = bucket.list_blobs(prefix=prefix)
  for blob in blobs:
    blob_uri = f"gs://{bucket_name}/{blob.name}"
    yield blob_uri


@dataclasses.dataclass
class InputConfig:
  bucket_name: str
  prefix: str
  batch_size: int
  patch_size: int
  sequence_length: int
  shuffle_buffer: int = 100_000


def get_inputs(config: InputConfig):
  generator_fn = functools.partial(gs_glob, config.bucket_name, config.prefix)

  dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
    generator_fn, output_signature=tf.TensorSpec(shape=(), dtype=tf.string, name=None))

  dataset = dataset.shuffle(config.shuffle_buffer)
  dataset = dataset.map(
    functools.partial(image_to_patch_sequences, patch_size=config.patch_size), 
    num_parallel_calls=128)
  # dataset = dataset.prefetch(128)
  dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
  dataset = dataset.batch(config.sequence_length)
  dataset = dataset.batch(config.batch_size)
  dataset = dataset.prefetch(1)

  yield from dataset.as_numpy_iterator()


if __name__ == '__main__':
  config = InputConfig(
    bucket_name='extension-screenshots-production',
    prefix='extensionScreenshots',
    batch_size=4,
    patch_size=256,
    sequence_length=128)

  inputs = get_inputs(config)
  for i, x in enumerate(inputs):
    print(x['patches'].shape)
    if i == 200:
      break
