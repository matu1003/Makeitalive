import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from data.video_utils import preprocess_frame, is_scene_change

def extract_pairs_from_video(
    video_path: str, 
    output_dir: str, 
    sample_every_n_seconds: float = 5.0,
    frame_gap: int = 3,
    target_size: int = 512,
    max_pairs: int = -1,
    clean: bool = False,
):
    """
    Reads a local video file and extracts image pairs (I_t, I_{t+k})
    for training.
    """
    if not os.path.exists(video_path):
        print(f"Error: file {video_path} does not exist.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    dir_A = out_path / "img_A"
    dir_B = out_path / "img_B"
    dir_A.mkdir(exist_ok=True)
    dir_B.mkdir(exist_ok=True)

    print(f"Opening local file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: unable to open the video file.")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30
    print(f"Detected FPS: {fps_video:.2f} frames/second")

    frame_interval = int(fps_video * sample_every_n_seconds)
    
    pbar = tqdm(desc="Extracted pairs", unit="pairs")
    
    pair_count = 0
    current_frame_idx = 0
    deleted_pairs = 0
    
    while True:
        ret_A, frame_A = cap.read()
        if not ret_A:
            break
            
        if current_frame_idx % frame_interval == 0:
            frame_B = None
            for _ in range(frame_gap):
                ret_B, temp_B = cap.read()
                current_frame_idx += 1
                if not ret_B:
                    break
                frame_B = temp_B
                
            if frame_B is None:
                break
                
            img_A = preprocess_frame(frame_A, target_size)
            img_B = preprocess_frame(frame_B, target_size)
            
            if clean:
                if is_scene_change(img_A, img_B):
                    deleted_pairs += 1
                    continue
            
            filename = f"pair_{pair_count:06d}.jpg"
            cv2.imwrite(str(dir_A / filename), img_A)
            cv2.imwrite(str(dir_B / filename), img_B)
            
            pair_count += 1
            pbar.update(1)
            
            if max_pairs > 0 and pair_count >= max_pairs:
                print(f"\nReached the limit of {max_pairs} pairs.")
                break
        else:
            current_frame_idx += 1
            
    cap.release()
    pbar.close()
    print(f"Extraction done! {pair_count} pairs saved to {output_dir}.")
    if clean:
        print(f"{deleted_pairs} pairs filtered out (scene changes detected).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a dataset of image pairs from a local video")
    parser.add_argument("--video", type=str, required=True, help="Path to the local video file (.mp4, .mkv, etc.)")
    parser.add_argument("--name", type=str, default="local_dataset", help="Dataset name (creates a ./data/NAME folder)")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between two extracted pairs")
    parser.add_argument("--gap", type=int, default=3, help="Frame gap (e.g. 3 = 3rd frame after A)")
    parser.add_argument("--size", type=int, default=512, help="Output image size (square)")
    parser.add_argument("--max_pairs", type=int, default=-1, help="Max number of pairs (-1 = all)")
    parser.add_argument("--clean", action="store_true", help="Filter out pairs that contain a scene change")    
    args = parser.parse_args()
    
    out_dir = os.path.join(".", "data", args.name)
    print(f"Parameters: Video={args.video}, Output={out_dir}, Interval={args.interval}s, Gap={args.gap} frames, Size={args.size}px")
    
    extract_pairs_from_video(
        video_path=args.video,
        output_dir=out_dir,
        sample_every_n_seconds=args.interval,
        frame_gap=args.gap,
        target_size=args.size,
        max_pairs=args.max_pairs,
        clean=args.clean,
    )
