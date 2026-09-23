"""
Animates a static picture with a trained Motion Flow U-Net.

The model predicts a single flow field f from the input image, with the training
convention warp(I_t, f)(x) = I_t(x + f(x)) ~ I_{t+k}. The animation is
obtained by warping the image with a progressively amplified version of this flow
(or, in autoregressive mode, by re-predicting the flow on each generated frame).

Usage:
    uv run src/motion_flow/infer.py \
        --checkpoint ./checkpoints/run_<timestamp>/model_best.pth \
        --image path/to/picture.jpg \
        --out ./outputs/landscape.gif
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image

from data.video_utils import preprocess_frame
from motion_flow.model import MotionFlowUNet
from motion_flow.warp import warp


def find_latest_checkpoint(ckpt_dir: str = "./checkpoints") -> str | None:
    """Returns model_best.pth (or model_latest.pth) of the most recent run, if any."""
    for run in sorted(glob.glob(os.path.join(ckpt_dir, "run_*")), reverse=True):
        for name in ("model_best.pth", "model_latest.pth"):
            path = os.path.join(run, name)
            if os.path.exists(path):
                return path
    return None


def load_model(checkpoint: str | None, device: torch.device) -> MotionFlowUNet:
    """Builds the U-Net and loads its weights. Without checkpoint, weights stay random."""
    model = MotionFlowUNet().to(device)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def load_image(path: str, size: int = 512) -> torch.Tensor:
    """Loads a picture, resizes + center-crops it to size x size, returns a (1, 3, H, W) tensor in [0, 1]."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(preprocess_frame(bgr, size), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


@torch.no_grad()
def predict_flow(model: MotionFlowUNet, img: torch.Tensor) -> torch.Tensor:
    """Predicts the (1, 2, H, W) flow field (dx, dy in pixels) of an image."""
    device = next(model.parameters()).device
    return model(img.to(device)).cpu()


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) tensor in [0, 1] -> (H, W, 3) uint8 array."""
    return (img[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def animate(img: torch.Tensor, flow: torch.Tensor, num_frames: int = 60,
            magnitude: float = 25.0, ping_pong: bool = False) -> list[np.ndarray]:
    """
    Warps the image with the flow scaled from 0 to `magnitude`
    (or back and forth with `ping_pong`). Returns RGB uint8 frames.
    """
    frames = []
    for i in range(num_frames):
        t = i / num_frames
        amplitude = np.sin(t * 2 * np.pi) if ping_pong else t
        frames.append(to_uint8(warp(img, flow * amplitude * magnitude)))
    return frames


@torch.no_grad()
def animate_autoregressive(model: MotionFlowUNet, img: torch.Tensor, steps: int = 30,
                           magnitude: float = 0.5) -> list[np.ndarray]:
    """
    Re-predicts the flow on each new frame and warps it again (experimental).
    Motion is not limited to a single flow field, but the image gets blurrier
    with each resampling step.
    """
    frames = [to_uint8(img)]
    current = img
    for _ in range(steps):
        flow = predict_flow(model, current)
        current = warp(current, flow * magnitude).clamp(0, 1)
        frames.append(to_uint8(current))
    return frames


def save_animation(frames: list[np.ndarray], path: str, fps: int = 20) -> None:
    """Saves RGB frames as a looping .gif, or as an .mp4 video."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith(".gif"):
        images = [Image.fromarray(f) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=int(1000 / fps), loop=0)
    else:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate a picture with a trained Motion Flow model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pth file (default: latest run in ./checkpoints)")
    parser.add_argument("--image", type=str, required=True, help="Input picture")
    parser.add_argument("--out", type=str, default="./outputs/animation.gif", help="Output .gif or .mp4")
    parser.add_argument("--size", type=int, default=512, help="Square working resolution (multiple of 16)")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames")
    parser.add_argument("--magnitude", type=float, default=25.0, help="Flow amplification at the last frame")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--ping_pong", action="store_true", help="Back-and-forth motion instead of a single sweep")
    parser.add_argument("--autoregressive", action="store_true", help="Re-predict the flow on every frame")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or find_latest_checkpoint()
    if checkpoint is None:
        parser.error("no checkpoint given and none found in ./checkpoints")
    print(f"Loading {checkpoint} on {device}")

    model = load_model(checkpoint, device)
    img = load_image(args.image, args.size)

    if args.autoregressive:
        frames = animate_autoregressive(model, img, steps=args.frames, magnitude=args.magnitude)
    else:
        frames = animate(img, predict_flow(model, img), args.frames, args.magnitude, args.ping_pong)

    save_animation(frames, args.out, args.fps)
    print(f"Saved {len(frames)} frames to {args.out}")
