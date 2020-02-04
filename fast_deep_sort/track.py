class Detection:
    def __init__(self, frame_idx, detection_box, features):
        self.frame = frame_idx
        self.detection_box = detection_box
        self.features = features

        
class Track:
    def __init__(self, first_detection):
        self.detections = []
        self.detections.append(first_detection)
        
    def is_expired(self, current_frame, expiration_threshold):
        return (current_frame - self.detections[-1].frame) > expiration_threshold
        
        
class TrackManager:
        
    def __init__(self, expiration_threshold=2):
        self.expiration_threshold = expiration_threshold
        self.active_tracks = []
        self.all_tracks = []
        
    def new_track(self, detection):
        track = Track(detection)
        self.all_tracks.append(track)
        return track
        
    def update_tracks(self,frame_idx, detections):
        found_tracks = []
        unassigned_tracks = self.active_tracks
        occluded_tracks = []
        detections = [Detection(frame_idx, detection_box, None) for detection_box in detections]
        
        for detection in detections:
            track:Track = None
                
            # track matching algorithm - currently random #TODO use actual algorithm
            if len(unassigned_tracks)>0:
                track = unassigned_tracks.pop()
                track.detections.append(detection)
                
            # if no match found, make new Track
            if track is None:
                track = self.new_track(detection)
                
            found_tracks.append(track)
            
        # only keep unassigned tracks that haven't expired in active_tracks
        for track in unassigned_tracks:
            if not track.is_expired(frame_idx, self.expiration_threshold):
                occluded_tracks.append(track)
        self.active_tracks = occluded_tracks + found_tracks

        return found_tracks