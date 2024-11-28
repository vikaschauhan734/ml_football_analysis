from ultralytics import YOLO
import supervision as sv



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

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection[0].names  # dictionary form key is 0, 1, 2 and values are player, goalkeeper, ball etc
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            print(detection_supervision)