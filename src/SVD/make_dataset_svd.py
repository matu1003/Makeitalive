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


def preprocess_frame(frame: np.ndarray, target_size: int) -> np.ndarray:
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
    diff = np.mean(np.abs(frame_a.astype(float) - frame_b.astype(float)))
    return diff > threshold


def compute_motion_score(frames: list[np.ndarray]) -> float:
    """
    Calcule le score de mouvement moyen entre frames consécutives.
    Un score élevé = beaucoup de mouvement (déplacement de caméra, vent, etc.)
    """
    diffs = []
    for i in range(len(frames) - 1):
        # Convertir en niveaux de gris pour comparer le mouvement structural
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
    sample_every_n_seconds: float = 0.5,  # interval court pour beaucoup de clips
    frame_gap: int = 25,                   # gap large pour capturer le mouvement drone
    max_clips: int = 5000,
    scene_cut_threshold: float = 30.0,
    motion_min: float = 3.0,              # score minimum — rejette les scènes statiques
    motion_max: float = 40.0,             # score maximum — rejette les cuts/glitches
):
    """
    Extrait des clips avec filtrage par niveau de mouvement.

    La logique "accéléré" : on échantillonne les frames avec un grand frame_gap
    (ex: 25 frames = ~1s d'écart à 24fps) pour capturer le mouvement de caméra drone.
    Le clip de 14 frames résultant représente ~25s de vidéo source compressé en 2s.
    SVD apprend ainsi à générer du mouvement prononcé de façon fluide.

    motion_min=3.0 : rejette les plans fixes (montagne sans vent, etc.)
    motion_max=40.0 : rejette les cuts durs et les transitions
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tmp_video = str(out_path / "tmp_landscape.mp4")
    print(f"Downloading video...")
    local_path, video_fps = download_video(youtube_url, tmp_video)

    frame_stride = max(1, round(video_fps / fps))
    # Avec frame_gap=25, on saute 25 frames source entre chaque frame du clip
    # À 24fps source: 25 frames = ~1s de mouvement réel par frame du clip
    frames_to_read = clip_len * frame_gap
    clip_start_interval = int(video_fps * sample_every_n_seconds)

    print(f"Video FPS: {video_fps:.1f}")
    print(f"Frame gap: {frame_gap} frames source par frame de clip")
    print(f"Mouvement capturé par clip: {clip_len * frame_gap / video_fps:.1f}s de vidéo source")
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
            # Collecter clip_len frames avec frame_gap frames source entre chaque
            clip_frames = [preprocess_frame(frame, target_size)]
            prev_raw = frame
            local_idx = 1
            scene_cut_detected = False

            while local_idx < clip_len:
                # Sauter frame_gap-1 frames
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

            # Filtrage par mouvement
            motion_score = compute_motion_score(clip_frames)

            if motion_score < motion_min:
                rejected_static += 1
                frame_idx += 1
                continue

            if motion_score > motion_max:
                rejected_cut += 1
                frame_idx += 1
                continue

            # Sauvegarder le clip
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
    print(f"Rejected — static: {rejected_static} | scene cuts: {rejected_cut}")
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