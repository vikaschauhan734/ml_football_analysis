import pickle
import cv2
import numpy as np
from src.utils.bbox_utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        self.minimum_distance = 5 # Minimum distance for camera movement is 5 pixels

        self.lk_params = dict(
            winSize = (15, 15), # Search Window size
            maxLevel = 2, # Downscale the image to find the large features; Downscale upto twice
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) # Stopping criteria
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1 # First two rows of pixels
        mask_features[:,900:1050] = 1 # Last rows of pixels

        self.features = dict(
            maxCorners = 100, # This is the maximum amount of corners we need to utilize for the good features
            qualityLevel = 0.3,
            minDistance = 3, # Minimum distance between the features
            blockSize = 7, # Search size of the feature
            mask = mask_features,
        )
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub

        camera_movement = [[0,0]*len(frames)] # [x,y]

        # Covering image to gray
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features) # Track corner features

        # Looping over each frame but starting from 2nd frame because 1st frame we already used
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features,_,_ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new, old) in enumerate(new_features, old_features):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        return camera_movement
