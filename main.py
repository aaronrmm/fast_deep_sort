import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow import keras
from importlib import import_module
from fast_deep_sort.preprocess import frame_batch_iterator
from fast_deep_sort.detection import load_detector
from fast_deep_sort.track import TrackManager


FRAMES_DIR = Path('input_frames')
IMAGE_FORMATS = ['jpg']
DETECTOR_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
BATCH_SIZE = 64
DETECTION_THRESHOLD = 0.5
EXPIRATION_THRESHOLD=10

def generate_tracks(matchmaker):
    batch_iterator = frame_batch_iterator(FRAMES_DIR, IMAGE_FORMATS, BATCH_SIZE)
    detector = load_detector(DETECTOR_NAME)
    tracker = TrackManager(expiration_threshold = EXPIRATION_THRESHOLD)

    frames_processed = 0
    with tf.Session():
        while(True):
            try:
                next_batch = batch_iterator.get_next().eval()
                batch_size = next_batch.shape[0]
                
                detection_dict = detector.__call__(next_batch)
                detection_scores = detection_dict["scores"]
                num_detections = detection_dict["num_detections"]
                detection_boxes = detection_dict["boxes"]
                detection_classes = detection_dict["classes"]
                # filter detections by confidence
                mask = detection_scores > DETECTION_THRESHOLD
                # keep detections separated by frame
                detections_by_frame = [detection_boxes[i][mask[i]] for i in range(batch_size)]

                # assign tracks to the detections of each frame
                for frame_idx, detections in enumerate(detections_by_frame):
                    frame_idx = frame_idx + frames_processed
                    tracks = tracker.update_tracks(frame_idx, detections, matchmaker)
                frames_processed+=batch_size
                print("%d frames processed..."%(frames_processed))
            except tf.errors.OutOfRangeError:
                print("End of dataset")  # ==> "End of dataset"
                break
    return tracker.all_tracks
    
class Matcher:
    
    def match(self, detections, tracks):
        matches = {}
        
        unassigned_tracks = [track for track in tracks]
        
        for detection in detections:
            # track matching algorithm - currently random #TODO use actual algorithm
            if len(unassigned_tracks)>0:
                track = unassigned_tracks.pop()
                track.add_detection(detection)
                matches[detection]=track
                
        return matches
    
if __name__ == "__main__":
    matchmaker = Matcher()
    print(generate_tracks(matchmaker))