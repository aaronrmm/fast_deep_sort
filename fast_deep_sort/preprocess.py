import os
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf


def image_dir_batch_iterator(frames_dir, image_extensions=['jpg'], batch_size=4):
    batch_iterator = (
        load_frames(frames_dir, image_extensions)
        .batch(batch_size)
        .make_one_shot_iterator()
    )
    return batch_iterator

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
    
def video_file_batch_iterator(video_path, batch_size=4):
    batch_iterator = (
        load_video_frames(video_path)
        .batch(batch_size)
        .make_one_shot_iterator()
    )
    return batch_iterator


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    # Dataset.from_generator allows specification of uint8 precision expected by object detectors
    def gen(): #requires a generator
        frame_id = 1
        while True:
            ret,frame = cap.read()
            if ret is False or frame_id >=50:
                frame_id+=1
                break
            frame_id+=1
            yield frame
            
    if cap.isOpened(): 
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        # calculate shape based on first np image tensor
        images_shape = tf.TensorShape([
            height,
            width,
            3
        ])

        tf_images = tf.data.Dataset.from_generator(gen, output_types=tf.uint8, output_shapes=(images_shape))
        return tf_images
    else:
        raise Exception('Failed to open video at %s'%(video_path))