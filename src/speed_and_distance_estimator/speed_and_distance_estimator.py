

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5 # Calculating speed in between 5 frames.
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window, number_of_frames) # For not go out of bounds

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue # We skip the player which is not in last frame but present in first frame
                    