import mediapy as media
import numpy as np
import argparse

def convert_to_bgr(frames):
    """Convert a list of RGB frames to BGR format."""
    # Assuming 'frames' is a list of NumPy arrays with shape (height, width, 3)
    bgr_frames = [frame[..., ::-1] for frame in frames]
    return bgr_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RGB video frames to BGR format.")
    parser.add_argument("--iv", type=str, help="Path to the input RGB video file.")
    parser.add_argument("--ov", type=str, help="Path to save the output BGR video file.")
    args = parser.parse_args()

    # Load the video frames
    frames = media.read_video(args.iv)

    # Convert frames from RGB to BGR
    bgr_frames = convert_to_bgr(frames)

    # Save the converted frames as a new video
    media.write_video(args.ov, bgr_frames, fps=30)  # Adjust fps as needed
