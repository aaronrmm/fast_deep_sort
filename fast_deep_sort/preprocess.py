import os
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf


def load_frames(frames_dir, image_extensions):
    image_files = [filename for filename in os.listdir(frames_dir) if filename.split('.')[-1] in image_extensions]
    np_images = [np.array(Image.open(frames_dir/filename), dtype=np.uint8) for filename in image_files]
    
    # Dataset.from_generator allows specification of uint8 precision expected by object detectors
    def gen(): #requires a generator
        for i in np_images: 
            yield i
    
    # calculate shape based on first np image tensor
    images_shape = tf.TensorShape([
        np_images[0].shape[0],
        np_images[0].shape[1],
        np_images[0].shape[2]
    ])
    
    tf_images = tf.data.Dataset.from_generator(gen, output_types=tf.uint8, output_shapes=(images_shape))
    return tf_images

if __name__ == "__main__":
    IMAGE_FORMATS = ['jpg']
    
    frames_dir = Path('input_frames')
    tf_images = load_frames(frames_dir, IMAGE_FORMATS)
    print(tf_images._flat_shapes)