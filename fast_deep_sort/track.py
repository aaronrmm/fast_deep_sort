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
    def __init__(self, first_detection):
        self.detections = {}
        self.add_detection(first_detection)
        self.last_detection = first_detection
        
    def add_detection(self, detection):
        self.detections[detection.frame_idx] = detection
        self.last_detection = detection
        
    def is_expired(self, current_frame, expiration_threshold):
        return (current_frame - self.get_last_detection().frame_idx) > expiration_threshold
        
    def get_last_detection(self):
        return self.last_detection
        
        
class TrackManager:
        
    def __init__(self, expiration_threshold=2):
        self.expiration_threshold = expiration_threshold
        self.active_tracks = []
        self.all_tracks = []
        
    def new_track(self, detection):
        track = Track(detection)
        self.all_tracks.append(track)
        return track
        
    def update_tracks(self,frame_idx, detections, matchmaker):
        found_tracks = []
        unassigned_tracks = self.active_tracks
        occluded_tracks = []
        detections = [Detection(frame_idx, detection_box, None) for detection_box in detections]
        matches: dict = matchmaker.match(detections, self.all_tracks)
        for detection in detections:
            track:Track = None
                
            # if no match found, make new Track
            if not detection in matches.keys():
                track = self.new_track(detection)
            else:
                track = matches[detection]
                
            found_tracks.append(track)
            
        # only keep unassigned tracks that haven't expired in active_tracks
        for track in unassigned_tracks:
            if not track.is_expired(frame_idx, self.expiration_threshold):
                occluded_tracks.append(track)
        self.active_tracks = occluded_tracks + found_tracks

        return found_tracks