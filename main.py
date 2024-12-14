from src.utils.video_utils import read_video, save_video
from src.trackers.tracker import tracker
import cv2
from src.team_assigner import TeamAssigner

def main():
    # Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #Initialize tracker
    tracke = tracker('models/best_11l.pt')

    tracks = tracke.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

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

    # Assign Player Teams
    team_assign = TeamAssigner()
    team_assign.assign_team_color(video_frames[0], tracks['players'][0]) # first frame and tracks only of first frame

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assign.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team # Assigning team
            tracks['players'][frame_num][player_id]['team_color'] = team_assign.team_colors[team] # Assigning team color

    # Draw output
    ## Draw object tracks
    output_video_frames = tracke.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_video/output_video.avi')

if __name__ == '__main__':
    main()