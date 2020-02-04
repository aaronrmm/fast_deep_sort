import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow import keras
from importlib import import_module
from fast_deep_sort.preprocess import load_frames
from fast_deep_sort.detection import load_detector
from fast_deep_sort.track import TrackManager


FRAMES_DIR = Path('input_frames')
IMAGE_FORMATS = ['jpg']
DETECTOR_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
BATCH_SIZE = 4
DETECTION_THRESHOLD = 0.7

def generate_tracks():
    tf_images = load_frames(FRAMES_DIR, IMAGE_FORMATS)
    detector = load_detector(DETECTOR_NAME)
    batch = tf_images.batch(4)
    detection_dict = detector(next(iter(batch))) #detection scores, num_detections
    detection_scores = detection_dict["detection_scores"]
    num_detections = detection_dict["num_detections"]
    detection_boxes = detection_dict["detection_boxes"]
    # filter detections by confidence
    mask = detection_scores > DETECTION_THRESHOLD
    # keep detections separated by frame
    detections_by_frame = [detection_boxes[i][mask[i]] for i in range(detection_boxes.shape[0])]
    
    # assign tracks to the detections of each frame
    tracker = TrackManager(expiration_threshold = 2)
    for frame_idx, detections in enumerate(detections_by_frame):
        tracks = tracker.update_tracks(frame_idx, detections)
    return tracks
    
if __name__ == "__main__":
    print(generate_tracks())