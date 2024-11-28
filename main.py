from src.utils.video_utils import read_video, save_video
from src.trackers.tracker import tracker

def main():
    # Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #Initialize tracker
    tracke = tracker('models/best_11l.pt')

    tracks = tracke.get_object_tracks(video_frames)

    # Save video
    save_video(video_frames, 'output_video/output_video.avi')

if __name__ == '__main__':
    main()