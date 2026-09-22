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
        --max_clips 5000
"""

import os
import cv2
import yt_dlp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import subprocess

from data.video_utils import preprocess_frame


def download_video(youtube_url: str, output_path: str) -> tuple[str, float]:
    if os.path.exists(output_path):
        print(f"Video already exists at {output_path}, skipping download.")
    else:
        cmd = [
            'yt-dlp',
            '-f', '136',
            '--retries', '20',
            '--fragment-retries', '20',
            '--concurrent-fragments', '4',
            '--buffer-size', '16K',
            '--http-chunk-size', '10M',
            '-o', output_path,
            youtube_url,
        ]
        subprocess.run(cmd, check=True)

    ydl_opts = {'format': '136', 'quiet': True, 'noplaylist': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        fps = float(info.get('fps') or 24)
    return output_path, fps


def is_scene_cut(frame_a: np.ndarray, frame_b: np.ndarray, threshold: float = 30.0) -> bool:
    diff = np.mean(np.abs(frame_a.astype(float) - frame_b.astype(float)))
    return diff > threshold


def compute_motion_score(frames: list[np.ndarray]) -> float:
    """
    Computes the mean motion score between consecutive frames.
    A high score = lots of motion (camera movement, wind, etc.)
    """
    diffs = []
    for i in range(len(frames) - 1):
        # Convert to grayscale to compare structural motion
        gray_a = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
        gray_b = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
        diff = np.mean(np.abs(gray_a - gray_b))
        diffs.append(diff)
    return float(np.mean(diffs))


def extract_clips(
    youtube_url: str,
    output_dir: str,
    clip_len: int = 14,
    fps: int = 7,
    target_size: int = 512,
    sample_every_n_seconds: float = 0.5,  # short interval to get many clips
    frame_gap: int = 25,                   # large gap to capture drone motion
    max_clips: int = 5000,
    scene_cut_threshold: float = 30.0,
    motion_min: float = 3.0,              # minimum score, rejects static scenes
    motion_max: float = 40.0,             # maximum score, rejects cuts/glitches
):
    """
    Extracts clips filtered by their amount of motion.

    "Time-lapse" logic: frames are sampled with a large frame_gap
    (e.g. 25 frames = ~1s apart at 24fps) to capture the drone camera motion.
    The resulting 14-frame clip covers ~25s of source video compressed into 2s,
    so SVD learns to generate pronounced yet smooth motion.

    motion_min=3.0: rejects static shots (mountain without wind, etc.)
    motion_max=40.0: rejects hard cuts and transitions
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tmp_video = str(out_path / "tmp_landscape.mp4")
    print(f"Downloading video...")
    local_path, video_fps = download_video(youtube_url, tmp_video)

    frame_stride = max(1, round(video_fps / fps))
    # With frame_gap=25, 25 source frames are skipped between two clip frames
    # At 24fps source: 25 frames = ~1s of real motion per clip frame
    frames_to_read = clip_len * frame_gap
    clip_start_interval = int(video_fps * sample_every_n_seconds)

    print(f"Video FPS: {video_fps:.1f}")
    print(f"Frame gap: {frame_gap} source frames per clip frame")
    print(f"Motion captured per clip: {clip_len * frame_gap / video_fps:.1f}s of source video")
    print(f"Motion filter: [{motion_min:.1f}, {motion_max:.1f}]")

    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        print("Error: could not open video.")
        return

    clip_count = 0
    rejected_static = 0
    rejected_cut = 0
    frame_idx = 0
    pbar = tqdm(desc="Clips extracted", unit="clips")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % clip_start_interval == 0:
            # Collect clip_len frames with frame_gap source frames between each
            clip_frames = [preprocess_frame(frame, target_size)]
            prev_raw = frame
            local_idx = 1
            scene_cut_detected = False

            while local_idx < clip_len:
                # Skip frame_gap-1 frames
                for _ in range(frame_gap - 1):
                    ret2, _ = cap.read()
                    frame_idx += 1
                    if not ret2:
                        break

                ret2, next_frame = cap.read()
                frame_idx += 1
                if not ret2:
                    break

                if is_scene_cut(prev_raw, next_frame, scene_cut_threshold):
                    scene_cut_detected = True
                    break

                clip_frames.append(preprocess_frame(next_frame, target_size))
                prev_raw = next_frame
                local_idx += 1

            if scene_cut_detected or len(clip_frames) < clip_len:
                frame_idx += 2
                continue

            # Motion filtering
            motion_score = compute_motion_score(clip_frames)

            if motion_score < motion_min:
                rejected_static += 1
                frame_idx += 1
                continue

            if motion_score > motion_max:
                rejected_cut += 1
                frame_idx += 1
                continue

            # Save the clip
            clip_dir = out_path / f"clip_{clip_count:06d}"
            clip_dir.mkdir(exist_ok=True)
            for i, f in enumerate(clip_frames):
                cv2.imwrite(str(clip_dir / f"frame_{i:03d}.jpg"), f,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            clip_count += 1
            pbar.update(1)
            pbar.set_postfix({"motion": f"{motion_score:.1f}", "rejected_static": rejected_static})

            if max_clips > 0 and clip_count >= max_clips:
                print(f"\nReached max_clips={max_clips}.")
                break

        frame_idx += 1

    cap.release()
    pbar.close()
    print(f"\nDone. {clip_count} clips saved.")
    print(f"Rejected clips, static: {rejected_static} | scene cuts: {rejected_cut}")
    print(f"Each clip spans {clip_len * frame_gap / video_fps:.1f}s of source video")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",        type=str,   default="https://www.youtube.com/watch?v=AKeUssuu3Is")
    parser.add_argument("--out",        type=str,   default="./data/svd_landscape")
    parser.add_argument("--clip_len",   type=int,   default=14)
    parser.add_argument("--fps",        type=int,   default=7)
    parser.add_argument("--size",       type=int,   default=512)
    parser.add_argument("--interval",   type=float, default=0.5)
    parser.add_argument("--frame_gap",  type=int,   default=25)
    parser.add_argument("--max_clips",  type=int,   default=5000)
    parser.add_argument("--motion_min", type=float, default=3.0)
    parser.add_argument("--motion_max", type=float, default=40.0)
    parser.add_argument("--scene_cut_threshold", type=float, default=30.0)
    args = parser.parse_args()

    extract_clips(
        youtube_url=args.url,
        output_dir=args.out,
        clip_len=args.clip_len,
        fps=args.fps,
        target_size=args.size,
        sample_every_n_seconds=args.interval,
        frame_gap=args.frame_gap,
        max_clips=args.max_clips,
        scene_cut_threshold=args.scene_cut_threshold,
        motion_min=args.motion_min,
        motion_max=args.motion_max,
    )