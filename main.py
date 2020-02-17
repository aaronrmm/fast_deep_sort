import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow import keras
from importlib import import_module
from fast_deep_sort.preprocess import image_dir_batch_iterator
from fast_deep_sort.detection import load_detector
from fast_deep_sort.track import TrackManager
from fast_deep_sort.distance_metrics import pairwise_cosine_distance as distance_metric 


FRAMES_DIR = Path('input_frames')
IMAGE_FORMATS = ['jpg']
DETECTOR_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
BATCH_SIZE = 12
DETECTION_THRESHOLD = 0.5
EXPIRATION_THRESHOLD=10
EMBEDDING_DIM = 128
INFINITE_DISTANCE = 1000
CHECKPOINT_PATH = Path('./triplet_reid/market1501_weights')
FEATURE_EXTRACTOR_PATH = './triplet_reid/encoder_trinet.pb'
INPUT_HEIGHT = 256
INPUT_WIDTH = 128
SIZE_ACTIVE_TRACK_BUFFER = 32
DEVICE = '/gpu:0'

def generate_tracks():

    frames_processed = 0
    # check memory usage of model with session config
    config = tf.ConfigProto(log_device_placement=False)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True
    with tf.Session(graph=tf.Graph(), config=config) as sess:
    
        batch_iterator = image_dir_batch_iterator(FRAMES_DIR, IMAGE_FORMATS, BATCH_SIZE)
        with tf.device(DEVICE):
            detector = load_detector(DETECTOR_NAME)
        tracker = TrackManager(max_tracks = SIZE_ACTIVE_TRACK_BUFFER, feature_dim = EMBEDDING_DIM, infinite_distance = INFINITE_DISTANCE)
        
        #load weights for triplet_reid
        output_graph_def = tf.GraphDef()
        with open(FEATURE_EXTRACTOR_PATH, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name='')
        print('{} ops in the frozen graph.'.format(len(output_graph_def.node)))
        
        print("%d frames processed..."%(frames_processed))
        while(True):
            try:
                #get batch of frames
                image = batch_iterator.get_next()
                frame_batch = image.eval()
                batch_size = frame_batch.shape[0]
                
                #get detections
                detection_dict = detector.__call__(frame_batch)
                detection_scores = detection_dict["scores"]
                num_detections = detection_dict["num_detections"]
                bbox_batch = detection_dict["boxes"]
                class_batch = detection_dict["classes"]
                
                # filter detections by confidence
                mask = detection_scores > DETECTION_THRESHOLD
                max_detections = np.max(np.sum(mask, axis=1))
                class_batch = class_batch[:,:max_detections]
                bbox_batch = bbox_batch[:,:max_detections,:]

                distance, detection_features, tracks = tracker.update_tracks(sess, frame_batch, bbox_batch, class_batch, distance_metric)
                frames_processed+=batch_size
                print("%d frames processed..."%(frames_processed))
            except tf.errors.OutOfRangeError:
                print("End of dataset")  # ==> "End of dataset"
                break
    return tracker.all_tracks
    
    
if __name__ == "__main__":
    print(generate_tracks())