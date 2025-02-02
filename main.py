from src.utils.video_utils import read_video, save_video
from src.trackers.tracker import tracker
from src.team_assigner import TeamAssigner
from src.player_ball_assigner import PlayerBallAssigner
from src.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from src.view_transformer.view_transformer import ViewTransformer
from src.speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator
import numpy as np

def main():
    # Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #Initialize tracker
    tracke = tracker('models/best_11l1280.pt')

    tracks = tracke.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Get object positions
    tracke.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')

    # Get ajusted position to tracks
    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracke.interpolate_ball_positions(tracks["ball"])

    # # Save cropped image of a player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     # crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # save the cropped image
    #     cv2.imwrite(f'output_video/cropped_img.jpg', cropped_image)

    #     break

    # Speed and Distance Estimator
    speed_and_disance_estimator = SpeedAndDistanceEstimator()
    speed_and_disance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assign = TeamAssigner()
    team_assign.assign_team_color(video_frames[0], tracks['players'][0]) # first frame and tracks only of first frame

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assign.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team # Assigning team
            tracks['players'][frame_num][player_id]['team_color'] = team_assign.team_colors[team] # Assigning team color

    # Assign Ball Aquaistion to Player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1]) # Last one that had the ball
    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object tracks
    output_video_frames = tracke.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_disance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_video/output_video.avi')

if __name__ == '__main__':
    main()