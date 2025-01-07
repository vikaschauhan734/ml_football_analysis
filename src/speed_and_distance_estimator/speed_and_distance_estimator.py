from src.utils.bbox_utils import measure_distance

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5 # Calculating speed in between 5 frames.
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window, number_of_frames) # For not go out of bounds

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue # We skip the player which is not in last frame but present in first frame
                    

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    if object not in total_distance: # If player not in total_distance then create new key and value
                        total_distance[object] = {}

                    if track_id not in total_distance[object]: # Initialize diatance as 0 for all track_ids
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue