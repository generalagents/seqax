import tensorflow as tf
import numpy as np
import os
import jax

def make_patches(image, patch_size):
    """
    Divides an image into patches of a given size. Pads the right side of the image if necessary.
    
    Parameters:
    image (numpy.ndarray): The input image array.
    patch_size (tuple): A tuple (patch_height, patch_width) indicating the size of each patch.
    
    Returns:
    numpy.ndarray: A 2D array of patches.
    """
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    
    if img_width % patch_width != 0:
        padded_width = (img_width // patch_width + 1) * patch_width
        padding = padded_width - img_width
        image = np.pad(image, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)

    if img_height % patch_height != 0:
        padded_height = (img_height // patch_height + 1) * patch_height
        padding = padded_height - img_height
        image = np.pad(image, ((0, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)

    # Calculate the number of patches needed along each dimension
    num_patches_y = img_height // patch_height
    num_patches_x = img_width // patch_width
    
    patches = []
    for y in range(num_patches_y):
        row = []
        for x in range(num_patches_x):  # +1 to include the last padded patch if needed
            patch = image[y*patch_height:(y+1)*patch_height, x*patch_width:(x+1)*patch_width]
            row.append(patch)
        patches.append(np.array(row))
    
    patches = np.array(patches)
    return patches

# Function to create synthetic data
def create_synthetic_episode(patch_size, num_channels, image_newline: int = -1):
    num_images = np.random.randint(10)
    images = []
    all_actions = []
    for _ in range(num_images):
        width = np.random.randint(300, 1920)
        height = np.random.randint(300, 1920)
        image = np.random.randn(height, width, num_channels)
        images.append(image)

        num_actions = np.random.randint(0, 10)
        actions = [np.random.randint(0, 100) for _ in range(num_actions)]
        all_actions.append(actions)

    patches = []
    tokens = []
    indices = []
    for image, actions in zip(images, all_actions):
        image_patches = make_patches(image, (patch_size, patch_size))
        
        for y in range(image_patches.shape[0]):
            for x in range(image_patches.shape[1]):
                indices.append(len(indices) * 2)
                patches.append(image_patches[y, x])
            
            indices.append(len(indices) * 2 + 1)
            tokens.append(image_newline)
        
        for action in actions:
            indices.append(len(indices) * 2 + 1)
            tokens.append(action)

    patches = np.array(patches, dtype=np.float32)
    tokens = np.array(tokens, dtype=np.int64)
    indices = np.array(indices, dtype=np.int64)

    return patches, tokens, indices

# Function to serialize data into TFRecord format
def serialize_example(patches, tokens, indices):
    feature = {
        'patches': tf.train.Feature(float_list=tf.train.FloatList(value=patches.flatten())),
        'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens.flatten())),
        'indices': tf.train.Feature(int64_list=tf.train.Int64List(value=indices.flatten()))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Parameters for synthetic data
num_episodes = 10
patch_size = 256
num_channels = 3
output_dir = 'synthetic_tfrecords'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate and write synthetic data to TFRecord files
tfrecord_filename = os.path.join(output_dir, 'episodes.tfrecord')
with tf.io.TFRecordWriter(tfrecord_filename) as writer:
    for episode_idx in range(num_episodes):
        patches, tokens, indices = create_synthetic_episode(patch_size, num_channels)
        serialized_example = serialize_example(patches, tokens, indices)
    
        writer.write(serialized_example)

print(f"Generated {num_episodes} TFRecord files in {output_dir}")

# Define your feature description for TFRecord parsing
feature_description = {
    'patches': tf.io.FixedLenSequenceFeature([patch_size, patch_size, num_channels], dtype=tf.float32, allow_missing=True),
    'tokens': tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
    'indices': tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True)
}

# Function to parse a single TFRecord example
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

# Create random subseqeunces
def _create_random_subsequence(example, seq_length):
    patches = example['patches']
    tokens = example['tokens']
    indices = example['indices']
    
    num_indices = tf.shape(indices)[0]

    # Ensure there are enough indices to choose a subsequence
    assert_op = tf.debugging.assert_greater_equal(
        num_indices, seq_length, message="Sequence length is greater than the number of available indices")

    with tf.control_dependencies([assert_op]):
        num_indices = tf.identity(num_indices)
    
    # Select a random starting point for the subsequence
    start_idx = tf.random.uniform([], minval=0, maxval=num_indices - seq_length + 1, dtype=tf.int32)
    end_idx = start_idx + seq_length
    
    # Create the subsequence
    subseq_indices = indices[start_idx: end_idx]  # [0, 1, 3, 2, [4, 5, 7, 6]]
    
    # Find the indices pointing to patches and tokens
    patch_indices = tf.boolean_mask(subseq_indices // 2, subseq_indices % 2 == 0)
    token_indices = tf.boolean_mask(subseq_indices // 2, subseq_indices % 2 == 1)

    # Gather patches and tokens
    subseq_patches = tf.gather(patches, patch_indices)
    subseq_tokens = tf.gather(tokens, token_indices)

    # Create subseq_indices

    subseq_example = {
        'patches': subseq_patches,
        'tokens': subseq_tokens,
        'indices': subseq_indices,
    }
    
    return subseq_example

# Function to create the dataset
def create_dataset(tfrecord_paths, batch_size, seq_length):
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    def _apply_random_subseq(example):
        return _create_random_subsequence(example, seq_length)
    
    # dataset = dataset.map(_apply_random_subseq, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.batch(batch_size, drop_remainder=True)
    # dataset = dataset.shuffle(...)
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

# Example usage
tfrecord_paths = ['synthetic_tfrecords/episodes.tfrecord']
batch_size = 32
seq_length = 10

dataset = create_dataset(tfrecord_paths, batch_size, seq_length)

# Iterate through the dataset
for batch in dataset:
    print('indices', batch['indices'])
    print(jax.tree_map(lambda x: x.shape, batch))
