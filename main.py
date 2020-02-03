import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow import keras
from importlib import import_module
from fast_deep_sort.preprocess import load_frames
from fast_deep_sort.detection import load_detector


FRAMES_DIR = Path('input_frames')
IMAGE_FORMATS = ['jpg']
DETECTOR_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
BATCH_SIZE = 4


def generate_tracks():
    tf_images = load_frames(FRAMES_DIR, IMAGE_FORMATS)
    detector = load_detector(DETECTOR_NAME)
    batch = tf_images.batch(4)
    output = detector(next(iter(batch)))
    return output
    
if __name__ == "__main__":
    print(generate_tracks())