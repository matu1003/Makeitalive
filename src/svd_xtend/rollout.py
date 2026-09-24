"""
Animates the first frame of held-out clips with an SVD_Xtend fine-tuned checkpoint and builds,
for each clip, a single comparison video: the still input, the raw generation, the real clip,
then the generation slowed down and interpolated to 24 fps.

This is how the SVD result videos of this repository were produced.

Usage:
    uv run src/svd_xtend/rollout.py \
        --checkpoint_dir ./checkpoints/svd_xtend \
        --clips_dir ./data/svd_landscape \
        --out_dir ./outputs \
        --num_clips 5
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = 320
NUM_FRAMES = 14


def find_latest_checkpoint(checkpoint_dir: str) -> Path:
    """Returns the highest checkpoint-<step> folder, or checkpoint_dir itself if there is none."""
    base = Path(checkpoint_dir)
    steps = sorted((d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
                   key=lambda d: int(d.name.split("-")[1]))
    return steps[-1] if steps else base


def load_pipeline(checkpoint_dir: str, device: str = "cuda"):
    """Loads the base SVD pipeline with the fine-tuned UNet of the given checkpoint."""
    import torch
    from diffusers import StableVideoDiffusionPipeline, UNetSpatioTemporalConditionModel

    run = find_latest_checkpoint(checkpoint_dir)
    weights = run / "unet" / "diffusion_pytorch_model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"No UNet weights at {weights}")
    if weights.stat().st_size < 1_000_000:
        raise ValueError(f"UNet weights look corrupted: only {weights.stat().st_size:,} bytes "
                         f"(is the checkpoint-saving patch applied?)")
    print(f"Loading {run} ({weights.stat().st_size / 1e9:.2f} GB)")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    unet = UNetSpatioTemporalConditionModel.from_pretrained(str(run / "unet"), torch_dtype=dtype)
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", unet=unet, torch_dtype=dtype,
    )
    pipe.vae.to(dtype=dtype)
    pipe.to(device)
    return pipe


def generate(pipe, image: Image.Image, motion_bucket_id: int = 127) -> list[np.ndarray]:
    """Generates NUM_FRAMES RGB frames from a single conditioning picture."""
    import torch

    with torch.autocast(device_type=pipe.device.type, dtype=pipe.dtype):
        frames = pipe(
            image,
            num_frames=NUM_FRAMES,
            num_inference_steps=25,
            fps=7,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=0.02,
            decode_chunk_size=4,   # decode 4 frames at a time to save VRAM
        ).frames[0]
    return [np.array(f) for f in frames]


def load_clip(clip_dir: str, size: int = SIZE) -> list[np.ndarray]:
    """Loads the frames of a clip as RGB arrays resized to size x size."""
    files = sorted(Path(clip_dir).glob("frame_*.jpg"))[:NUM_FRAMES]
    return [np.array(Image.open(f).convert("RGB").resize((size, size))) for f in files]


def _write(frames: list[np.ndarray], path: str, fps: float) -> str:
    """Writes RGB frames as an mp4."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def _ffmpeg(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")


def _title(src: str, title: str, dst: str) -> str:
    """Burns a title into the top-left corner of a video."""
    _ffmpeg(["-i", src,
             "-vf", f"drawtext=text='{title}':fontsize=20:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.6",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", dst])
    return dst


def _titled(frames: list[np.ndarray], title: str, fps: float, dst: str) -> str:
    """Writes the frames as a video with their title burned in."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        _write(frames, tmp.name, fps)
        return _title(tmp.name, title, dst)


def build_comparison(input_frame: np.ndarray, generated: list[np.ndarray],
                     ground_truth: list[np.ndarray], output_path: str,
                     input_seconds: int = 3, fps: int = 7,
                     slow_fps: int = 2, final_fps: int = 24) -> str:
    """
    Concatenates INPUT, GENERATED, GROUND TRUTH and GENERATED + INTERP into one labelled video.
    The last segment is the generation slowed down to slow_fps to amplify the motion,
    then interpolated back to final_fps with ffmpeg minterpolate.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        parts = [
            _titled([input_frame] * (input_seconds * final_fps), "INPUT", final_fps, f"{tmp}/input.mp4"),
            _titled(generated, "GENERATED", fps, f"{tmp}/gen.mp4"),
            _titled(ground_truth, "GROUND TRUTH", fps, f"{tmp}/gt.mp4"),
        ]

        slow = _write(generated, f"{tmp}/slow.mp4", slow_fps)
        _ffmpeg(["-i", slow,
                 "-vf", f"minterpolate=fps={final_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18", f"{tmp}/interp.mp4"])
        parts.append(_title(f"{tmp}/interp.mp4", "GENERATED + INTERP", f"{tmp}/interp_t.mp4"))

        listing = Path(tmp) / "concat.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18", output_path])

    return output_path


def rollout(checkpoint_dir: str, clips_dir: str, out_dir: str, num_clips: int = 5,
            motion_bucket_id: int = 127, device: str = "cuda") -> list[str]:
    """Runs the comparison on the first num_clips clips of clips_dir."""
    pipe = load_pipeline(checkpoint_dir, device)
    clips = sorted(d for d in Path(clips_dir).iterdir() if d.is_dir() and d.name.startswith("clip_"))[:num_clips]

    outputs = []
    for i, clip in enumerate(clips):
        print(f"[{i + 1}/{len(clips)}] {clip.name}")
        ground_truth = load_clip(str(clip))
        image = Image.fromarray(ground_truth[0])
        generated = generate(pipe, image, motion_bucket_id)
        outputs.append(build_comparison(ground_truth[0], generated, ground_truth,
                                        f"{out_dir}/compare_{clip.name}.mp4"))
        print(f"  -> {outputs[-1]}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SVD_Xtend comparison videos")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Folder holding the checkpoint-<step> directories (or a checkpoint itself)")
    parser.add_argument("--clips_dir", type=str, required=True, help="Dataset of clips (clip_*/frame_*.jpg)")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--num_clips", type=int, default=5)
    parser.add_argument("--motion_bucket_id", type=int, default=127, help="0=static, 255=lots of motion")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rollout(args.checkpoint_dir, args.clips_dir, args.out_dir,
            args.num_clips, args.motion_bucket_id, args.device)
