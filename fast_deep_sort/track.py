import numpy as np
import tensorflow as tf


class Detection:
    def __init__(self, frame_idx, detection_box, features=None):
        self.frame_idx = frame_idx
        self.detection_box = detection_box
        
    def left(self, image_width):
        return image_width * self.detection_box[1].item()
        
    def right(self, image_width):
        return image_width * self.detection_box[3].item()
        
    def top(self, image_height):
        return image_height * self.detection_box[0].item()
        
    def bottom(self, image_height):
        return image_height * self.detection_box[2].item()
        
    def width(self, image_width):
        return self.right(image_width) - self.left(image_width)
        
    def height(self, image_height):
        return self.bottom(image_height) - self.top(image_height)
        
    def mask(height, width):
        mask = np.zeros((height, width, 3))
        mask[bottom(height):top(height),left(width):right(width)]=1
        return mask
        
        
class Track:
    def __init__(self, track_id, class_id, buffer_idx, frame_idx, initial_bbox):
        self.buffer_idx = buffer_idx # set to -1 when inactive
        self.track_id = track_id
        self.last_frame = frame_idx
        self.detections = {frame_idx:Detection(frame_idx, initial_bbox)}
        self.class_id = class_id

    def add_detection(self, frame_idx, bbox):
        self.detections[frame_idx] = Detection(frame_idx, bbox)
        self.last_frame = frame_idx

    def is_active(self):
        return self.buffer_index != -1
    
    def get_last_detection(self):
        return self.detections[self.last_frame]
    
    
class TrackManager:
    def __init__(self, max_tracks=32, feature_dim=128, infinite_distance= 1000):
        self.next_track_id = 0
        self.last_frame_idx = 0
        self.all_tracks = []
        self.active_tracks = []
        self.active_track_features = np.full((max_tracks, feature_dim), fill_value =infinite_distance, dtype=np.float32)
        self.usable_buffer_slots = list(range(max_tracks))
        
    def add_track(self, frame_idx, initial_bbox, features):
        # assign slot in buffer
        empty_buffer_idx = self.usable_buffer_slots.pop()
        track = Track(
            track_id = self.next_track_id,
            class_id = 0, # TODO support multiple classes
            buffer_idx = empty_buffer_idx,
            frame_idx = frame_idx,
            initial_bbox = initial_bbox)
        self.next_track_id+=1
        self.all_tracks.append(track)
        self.active_tracks.append(track)
        self.active_track_features[empty_buffer_idx] = features
    
    def update_tracks(self, sess, frame_batch, bbox_batch, class_batch, distance_metric, in_img, emb, infinite_distance= 1000):
        INPUT_HEIGHT = 256
        INPUT_WIDTH = 128
        in_img = sess.graph.get_tensor_by_name('input:0')
        print('frame batch',frame_batch.shape)
        emb = sess.graph.get_tensor_by_name('head/out_emb:0')
        for row_idx, frame_idx in enumerate(range(self.last_frame_idx, self.last_frame_idx+frame_batch.shape[0])):
            print('frame index', frame_idx)
            # resize detection images
            detection_images = resize_images(bbox_batch[row_idx], frame_batch[row_idx])
            detection_images = tf.image.resize_images(detection_images, [INPUT_HEIGHT, INPUT_WIDTH])
            # extract features -> np.array(num_detections, embedding_dim)
            new_detection_features = sess.run(emb, feed_dict={in_img: detection_images.eval()})
            distance = distance_metric(  # num_detections x num_tracks
                new_detection_features, # num_detections x embedding size
                self.active_track_features # num_tracks x embedding size
            )
            # find best detection for each active track
            for active_track in self.active_tracks:
                best_detection_index = np.argmin(distance[:, active_track.buffer_idx])
                
                # add detection to track
                active_track.add_detection(frame_idx, bbox_batch[row_idx, best_detection_index])
                active_track.last_frame = frame_idx #TODO this and previous step can be combined in setter
                
                # remove detection from distance metric
                distance[best_detection_index, :] = infinite_distance
                
            # create new tracks for unmatched detections
            for detection_idx, detection in enumerate(distance):
                if detection[0]!=infinite_distance: # is unmatched detection
                    self.add_track(
                        frame_idx = frame_idx,
                        initial_bbox= bbox_batch[row_idx, detection_idx],
                        features= new_detection_features[detection_idx])
        
        # TODO remove expired tracks from active track list and buffer
        expired_tracks = []
        # TODO get expired tracks
        for track in expired_tracks:
            self.active_tracks.remove(track)
            self.usable_buffer_slots.append(track.buffer_idx)
            self.active_track_features[track.buffer_idx,:] = infinite_distance
            track.buffer_idx = -1
        
        self.last_frame_idx += frame_batch.shape[0]
        return distance, new_detection_features, self.all_tracks # for debugging
        

def extract_crops(detection_boxes, image_numpy):
    image_height = image_numpy.shape[0]
    image_width = image_numpy.shape[1]
    image_channels = image_numpy.shape[2]
    h = np.arange(image_width)#*b.shape[2]).reshape(b.shape[1],b.shape[2])
    v = np.arange(image_height)
    mask = (detection_boxes[:,1, None]*image_width <=h) & (detection_boxes[:,3, None]*image_width >= h)
    mask = np.stack([mask for _ in range(image_height)], axis=1)
    mask[(detection_boxes[:,0, None]*image_height > v) | (detection_boxes[:,2, None]*image_height < v)]=False
    mask = np.stack([mask for _ in range(image_channels)], axis=3)
    return image_numpy*mask
resize_images = extract_crops