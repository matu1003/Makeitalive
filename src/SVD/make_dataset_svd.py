"""
make_dataset_svd.py
--------------------
Extracts short video clips from a YouTube video for SVD fine-tuning.
Designed to run on Google Colab.

Usage:
    python make_dataset_svd.py \
        --url "https://www.youtube.com/watch?v=AKeUssuu3Is" \
        --out "./data/svd_landscape" \
        --clip_len 14 \
        --fps 7 \
        --size 512 \
        --max_clips 2000

Install dependencies first:
    pip install yt-dlp opencv-python-headless tqdm
"""

import os
import cv2
import yt_dlp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def download_video(youtube_url: str, output_path: str) -> tuple[str, float]:
    """
    Downloads the video to a local file instead of streaming.
    More reliable on Colab than direct stream URLs.
    """
    ydl_opts = {
    'format': '136',
    'quiet': False,
    'noplaylist': True,
    'outtmpl': output_path,
    'retries': 10,
    'fragment_retries': 10,
    'file_access_retries': 10,
    'retry_sleep_functions': {'http': lambda n: 2 ** n},  # exponential backoff
    'concurrent_fragment_downloads': 4,
    'buffersize': 1024 * 16,
    'http_chunk_size': 10 * 1024 * 1024,  # 10MB chunks
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        fps = float(info.get('fps') or 24)
    return output_path, fps


def preprocess_frame(frame: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize so the short side = target_size, then center-crop to a square.
    """
    h, w = frame.shape[:2]
    if h < w:
        new_h, new_w = target_size, int(w * target_size / h)
    else:
        new_h, new_w = int(h * target_size / w), target_size

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    sy = (new_h - target_size) // 2
    sx = (new_w - target_size) // 2
    return resized[sy:sy + target_size, sx:sx + target_size]


def is_scene_cut(frame_a: np.ndarray, frame_b: np.ndarray, threshold: float = 30.0) -> bool:
    """
    Detects hard scene cuts by comparing mean absolute difference between two frames.
    Avoids saving clips that span a transition.
    """
    diff = np.mean(np.abs(frame_a.astype(float) - frame_b.astype(float)))
    return diff > threshold


def extract_clips(
    youtube_url: str,
    output_dir: str,
    clip_len: int = 14,          # number of frames per clip (SVD default is 14 or 25)
    fps: int = 7,                # target fps for training clips (lower = smoother apparent motion)
    target_size: int = 512,
    sample_every_n_seconds: float = 4.0,  # how often to start a new clip
    max_clips: int = 2000,
    scene_cut_threshold: float = 30.0,
):
    """
    Main extraction loop. For each sample point in the video, extracts a clip of
    `clip_len` frames at the native video fps, then saves them as an ordered
    sequence of JPEGs inside a numbered folder.

    Folder structure:
        output_dir/
            clip_000000/
                frame_000.jpg
                frame_001.jpg
                ...
                frame_013.jpg   (for clip_len=14)
            clip_000001/
                ...
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tmp_video = "/content/tmp_landscape.mp4"
    print(f"Downloading video to {tmp_video} ...")
    try:
        local_path, video_fps = download_video(youtube_url, tmp_video)
    except Exception as e:
        print(f"yt-dlp error: {e}")
        return

    print(f"Opening local video file...")
    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        print("Error: could not open video stream.")
        return

    clip_count = 0
    frame_idx = 0
    pbar = tqdm(desc="Clips extracted", unit="clips")

    frame_stride = max(1, round(video_fps / fps))
    frames_to_read = clip_len * frame_stride
    clip_start_interval = int(video_fps * sample_every_n_seconds)

    print(f"Video FPS: {video_fps:.1f} | Frame stride: {frame_stride} | Clip interval: {clip_start_interval} frames")
    
    while True:
        # --- Read one source frame ---
        ret, frame = cap.read()
        if not ret:
            break

        # Is this a clip start point?
        if frame_idx % clip_start_interval == 0:
            # Collect `clip_len` frames, striding through the source video
            clip_frames = [preprocess_frame(frame, target_size)]
            prev_frame = frame
            local_idx = 1

            while local_idx < clip_len:
                # Skip (frame_stride - 1) source frames to hit target fps
                for _ in range(frame_stride - 1):
                    ret2, _ = cap.read()
                    frame_idx += 1
                    if not ret2:
                        break

                ret2, next_frame = cap.read()
                frame_idx += 1
                if not ret2:
                    break

                # Scene cut detection — discard clip if transition detected
                if is_scene_cut(prev_frame, next_frame, scene_cut_threshold):
                    clip_frames = []   # signal: discard
                    break

                clip_frames.append(preprocess_frame(next_frame, target_size))
                prev_frame = next_frame
                local_idx += 1

            # Save only complete, clean clips
            if len(clip_frames) == clip_len:
                clip_dir = out_path / f"clip_{clip_count:06d}"
                clip_dir.mkdir(exist_ok=True)
                for i, f in enumerate(clip_frames):
                    cv2.imwrite(str(clip_dir / f"frame_{i:03d}.jpg"), f,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                clip_count += 1
                pbar.update(1)

            if max_clips > 0 and clip_count >= max_clips:
                print(f"\nReached max_clips={max_clips}. Stopping.")
                break

        frame_idx += 1

    cap.release()
    pbar.close()
    print(f"\nDone. {clip_count} clips saved to {output_dir}")
    print(f"Each clip: {clip_len} frames @ {fps} fps → {clip_len / fps:.1f}s of motion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SVD training clips from YouTube")
    parser.add_argument("--url",      type=str,   default="https://www.youtube.com/watch?v=AKeUssuu3Is")
    parser.add_argument("--out",      type=str,   default="./data/svd_landscape")
    parser.add_argument("--clip_len", type=int,   default=14,    help="Frames per clip (14 or 25 for SVD)")
    parser.add_argument("--fps",      type=int,   default=7,     help="Target fps of saved clips")
    parser.add_argument("--size",     type=int,   default=512,   help="Output image size (square)")
    parser.add_argument("--interval", type=float, default=4.0,   help="Seconds between clip start points")
    parser.add_argument("--max_clips",type=int,   default=2000,  help="Max clips to extract (-1 = all)")
    parser.add_argument("--scene_cut_threshold", type=float, default=30.0)
    args = parser.parse_args()

    extract_clips(
        youtube_url=args.url,
        output_dir=args.out,
        clip_len=args.clip_len,
        fps=args.fps,
        target_size=args.size,
        sample_every_n_seconds=args.interval,
        max_clips=args.max_clips,
        scene_cut_threshold=args.scene_cut_threshold,
    )
