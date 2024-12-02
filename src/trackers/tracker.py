from ultralytics import YOLO
import supervision as sv
import pickle
import os



class tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20 # 20 frames in one loop to solve memory issue problem
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1) # confidence is 0.1: predict only when confidence is greater than 0.1
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        def add_track_data(tracks, category, frame_num, track_id, bbox):
            while len(tracks[category]) <= frame_num:
                tracks[category].append({})
            tracks[category][frame_num][track_id] = {"bbox": bbox}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # dictionary form key is 0, 1, 2, 3 and values are ball, goalkeeper, player, referee
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper class to player class
            for obj_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[obj_idx] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Appending a dictionary player track id as key and bounding box list as value 
            #tracks["players"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    add_track_data(tracks,"players",frame_num,track_id, bbox)

                if cls_id == cls_names_inv["referee"]:
                    add_track_data(tracks,"referees",frame_num,track_id, bbox)
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    add_track_data(tracks,"ball",frame_num,1, bbox)

            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)


            return tracks # dictionary of lists of dictionaries