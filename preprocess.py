import os
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf


def load_frames(frames_dir, image_extensions):
    image_files = [filename for filename in os.listdir(frames_dir) if filename.split('.')[-1] in image_extensions]
    np_images = [np.array(Image.open(frames_dir/filename)) for filename in image_files]
    tf_images = tf.data.Dataset.from_tensors(np_images)
    return tf_images

if __name__ == "__main__":
    IMAGE_FORMATS = ['jpg']
    
    frames_dir = Path('input_frames')
    tf_images = load_frames(frames_dir, IMAGE_FORMATS)
    print(tf_images._flat_shapes)