import os
import cv2
import yt_dlp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from data.video_utils import preprocess_frame, is_scene_change

def download_and_extract_pairs(
    youtube_url: str, 
    output_dir: str, 
    sample_every_n_seconds: float = 5.0,
    frame_gap: int = 3,
    target_size: int = 512,
    max_pairs: int = -1,
    clean: bool = False,
):
    """
    Streams a YouTube video (without downloading the whole file) and extracts
    image pairs (I_t, I_{t+k}) for motion flow training.
    
    Args:
        youtube_url: URL of the video (a local video path also works).
        output_dir: folder where the pairs are saved.
        sample_every_n_seconds: time interval between two samples (in seconds).
        frame_gap: number of frames between frame A and frame B.
        target_size: size of the (square) extracted images (e.g. 512).
        max_pairs: maximum number of pairs to extract, -1 to extract everything.
        clean: if True, drop the pairs that contain a scene change.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Folders for the start frame (img_A) and the end frame (img_B)
    dir_A = out_path / "img_A"
    dir_B = out_path / "img_B"
    dir_A.mkdir(exist_ok=True)
    dir_B.mkdir(exist_ok=True)

    if os.path.exists(youtube_url):
        print(f"Using local file: {youtube_url}")
        stream_url = youtube_url
        cap = cv2.VideoCapture(stream_url)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        if fps_video <= 0:
            fps_video = 30
        print(f"Opening local file (FPS: {fps_video})...")
    else:
        print(f"Fetching stream URL with yt-dlp for {youtube_url}...")
        
        # yt-dlp config to get the direct stream URL (avoids downloading the full video)
        ydl_opts = {
            'format': 'best[ext=mp4]/best', # Prefer mp4 for OpenCV compatibility
            'quiet': True,
            'noplaylist': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                stream_url = info['url']
                fps_video = info.get('fps', 30)
                if fps_video is None:
                    fps_video = 30
        except Exception as e:
            print(f"Error during yt-dlp extraction: {e}")
            return

        print(f"Opening video stream (estimated FPS: {fps_video})...")
        cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Error: unable to open the video stream.")
        return

    # Sampling interval in number of frames
    frame_interval = int(fps_video * sample_every_n_seconds)
    
    # Progress bar
    pbar = tqdm(desc="Extracted pairs", unit="pairs")
    
    pair_count = 0
    current_frame_idx = 0
    deleted_pairs = 0
    
    while True:
        # Read the current frame (Frame A)
        ret_A, frame_A = cap.read()
        if not ret_A:
            break
            
        # Only keep the frames that fall on the sampling interval
        if current_frame_idx % frame_interval == 0:
            # Look for frame B (Frame A + frame_gap)
            frame_B = None
            
            # Move the cursor forward up to frame B
            for _ in range(frame_gap):
                ret_B, temp_B = cap.read()
                current_frame_idx += 1
                if not ret_B:
                    break
                frame_B = temp_B
                
            if frame_B is None:
                break # End of the video reached during the gap
                
            # Process both frames
            # 1. Resize and center crop to get a perfect square
            img_A = preprocess_frame(frame_A, target_size)
            img_B = preprocess_frame(frame_B, target_size)
            
            # 2. Scene-change filtering if enabled
            if clean:
                if is_scene_change(img_A, img_B):
                    # Frames too different, skip the pair
                    deleted_pairs += 1
                    continue
            
            # Save both images as JPEG
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
    print(f"Extraction done! {pair_count} pairs saved to {output_dir}. {deleted_pairs} pairs filtered out.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a dataset of image pairs from a YouTube video")
    parser.add_argument("--url", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--name", type=str, default="youtube_landscape", help="Dataset name (creates a ./data/NAME folder)")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between two extracted pairs")
    parser.add_argument("--gap", type=int, default=3, help="Frame gap (e.g. 3 = 3rd frame after A)")
    parser.add_argument("--size", type=int, default=512, help="Output image size (square)")
    parser.add_argument("--max_pairs", type=int, default=-1, help="Max number of pairs (-1 = all)")
    parser.add_argument("--clean", action="store_true", help="Filter out pairs that contain a scene change")    
    args = parser.parse_args()
    
    out_dir = os.path.join(".", "data", args.name)
    print(f"Parameters: URL={args.url}, Output={out_dir}, Interval={args.interval}s, Gap={args.gap} frames, Size={args.size}px, Max Pairs={args.max_pairs}, Clean={args.clean}")
    download_and_extract_pairs(
        youtube_url=args.url,
        output_dir=out_dir,
        sample_every_n_seconds=args.interval,
        frame_gap=args.gap,
        target_size=args.size,
        max_pairs=args.max_pairs,
        clean=args.clean,
    )
