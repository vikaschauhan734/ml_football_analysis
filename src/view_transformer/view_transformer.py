import numpy as np
import cv2


class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_verticies = np.array({
            [110, 1035], # Points of the trapeziod
            [265, 275],
            [910,260],
            [1640,915]
        })

        self.target_verticies = np.array({
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        })

        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)

    def transform_point(self, point):
        pass

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed