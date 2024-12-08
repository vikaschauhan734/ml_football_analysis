from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
from src.utils.bbox_utils import get_bbox_width, get_center_of_bbox


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
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)


        return tracks # dictionary of lists of dictionaries
    
    def draw_ellipse(self, frame, bbox, color, track_id=None): # Drawing ellipse
        y2 = int(bbox[3]) # y2 is the bottom
        x_center,_ = get_center_of_bbox(bbox) # center of the x axis
        width = get_bbox_width(bbox) # Width of ellipse

        cv2.ellipse(frame,
                    center=(x_center, y2),
                    axes=(int(width), int(0.35*width)), # minor axis will be 35% of major axis.
                    angle=0.0,
                    startAngle=-45, # ellipse drawing will start from 45 degrees
                    endAngle=235,   # and end before 235 degrees
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4
                    )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2 # Top left corner of the rectangle
        x2_rect = x_center + rectangle_width//2 # Bottom right corner of the rectangle
        y1_rect = (y2 - rectangle_height//2) + 15 # Just random buffer 
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED # Filled Rectangle
                          )
            x1_text = x1_rect + 12
            y1_text = y1_rect + 15
            if track_id > 99:
                x1_text -= 10 

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_text)),
                cv2.FONT_HERSHEY_SIMPLEX, # Font type
                0.6, # Font ratio
                (0,0,0), # Black Color
                2 # Thickness
            )
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                framee = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id) # (0, 0, 255) is the red color in BGR format.

            # Draw referees
            for _ , referee in referee_dict.items():
                framee = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255)) # (0, 255, 255) is the yellow color in BGR format.

            output_video_frames.append(framee)
        return output_video_frames
