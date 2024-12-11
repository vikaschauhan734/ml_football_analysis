from src.utils.video_utils import read_video, save_video
from src.trackers.tracker import tracker
import cv2

def main():
    # Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #Initialize tracker
    tracke = tracker('models/best_11l.pt')

    tracks = tracke.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # # Save cropped image of a player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     # crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # save the cropped image
    #     cv2.imwrite(f'output_video/cropped_img.jpg', cropped_image)

    #     break

    # Draw output
    ## Draw object tracks
    output_video_frames = tracke.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_video/output_video.avi')

if __name__ == '__main__':
    main()