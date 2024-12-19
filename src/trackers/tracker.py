from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import pandas as pd
import os
import cv2
from src.utils.bbox_utils import get_bbox_width, get_center_of_bbox


class tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions] # Get the data of track id = 1 if it is not present then give empty dictionary. And from that get bbox if not bbox then empty list
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate() 
        # If missing detection is first one than it will not interploate so we will replace with nearest detection
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


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

    def draw_triangle(self, frame, bbox, color): # Inverted triangle
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x,y],
                                    [x-10,y-20],
                                    [x+10,y-20]
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED) # Drawing filled triangle
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2) # Drawing border for triangle
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1) # -1 for filled
        alpha = 0.4 # For transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100: .2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100: .2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                framee = self.draw_ellipse(frame, player["bbox"], color, track_id) # (0, 0, 255) is the red color in BGR format.

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))

            # Draw referees
            for _ , referee in referee_dict.items():
                framee = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255)) # (0, 255, 255) is the yellow color in BGR format.

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0)) # Green color

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(framee)
        return output_video_frames
