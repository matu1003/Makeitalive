"""
train_svd_lora.py
------------------
Fine-tunes Stable Video Diffusion (SVD) temporal attention layers
using LoRA on a dataset of landscape video clips.

Designed to run on Google Colab with an A100 GPU.
Expected training time: ~1.5–2.5 hours for 2000 clips, 3 epochs.

Requirements:
    pip install diffusers[torch] transformers accelerate peft \
                opencv-python-headless tqdm Pillow

Usage:
    python train_svd_lora.py \
        --data_dir "./data/svd_landscape" \
        --output_dir "./checkpoints/svd_lora" \
        --epochs 3 \
        --batch_size 1 \
        --lora_rank 16
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import StableVideoDiffusionPipeline, EulerDiscreteScheduler
from diffusers.models import UNetSpatioTemporalConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class LandscapeClipDataset(Dataset):
    """
    Loads clips extracted by make_dataset_svd.py.

    Each clip is a folder of JPEGs: frame_000.jpg ... frame_013.jpg
    Returns:
        frames  : (T, 3, H, W) float32 tensor normalized to [-1, 1]
        cond    : (3, H, W)    first frame, normalized to [-1, 1]  (the conditioning image)
    """

    def __init__(self, data_dir: str, clip_len: int = 14, size: int = 512):
        self.clip_dirs = sorted([
            d for d in Path(data_dir).iterdir()
            if d.is_dir() and len(list(d.glob("frame_*.jpg"))) >= clip_len
        ])
        self.clip_len = clip_len
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),                        # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5],        # → [-1, 1]
                                  [0.5, 0.5, 0.5]),
        ])
        print(f"Found {len(self.clip_dirs)} clips in {data_dir}")

    def __len__(self):
        return len(self.clip_dirs)

    def __getitem__(self, idx):
        clip_dir = self.clip_dirs[idx]
        frame_paths = sorted(clip_dir.glob("frame_*.jpg"))[:self.clip_len]

        frames = torch.stack([
            self.transform(Image.open(p).convert("RGB"))
            for p in frame_paths
        ])  # (T, 3, H, W)

        cond_frame = frames[0]  # condition on the first frame
        return frames, cond_frame


# ─────────────────────────────────────────────
# LoRA config — targets ONLY temporal attention
# ─────────────────────────────────────────────

def get_lora_config(rank: int = 16, alpha: int = 16) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=[
            # Temporal attention uniquement (attn dans temporal_transformer_blocks)
            "temporal_transformer_blocks.0.attn1.to_q",
            "temporal_transformer_blocks.0.attn1.to_k",
            "temporal_transformer_blocks.0.attn1.to_v",
            "temporal_transformer_blocks.0.attn1.to_out.0",
            "temporal_transformer_blocks.0.attn2.to_q",
            "temporal_transformer_blocks.0.attn2.to_k",
            "temporal_transformer_blocks.0.attn2.to_v",
            "temporal_transformer_blocks.0.attn2.to_out.0",
        ],
        lora_dropout=0.05,
        bias="none",
    )


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # A100 handles bf16 natively — faster and more stable than fp16

    print(f"[{datetime.now():%H:%M:%S}] Device: {device} | dtype: {dtype}")
    print(f"[{datetime.now():%H:%M:%S}] Loading SVD pipeline...")

    # ── Load pipeline ──────────────────────────────────────────────────────────
    MODEL_ID = "stabilityai/stable-video-diffusion-img2vid"
    # Use img2vid (14 frames). For 25 frames use: stabilityai/stable-video-diffusion-img2vid-xt

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16",   # downloads the fp16 weights (~5GB vs 10GB)
    )

    unet: UNetSpatioTemporalConditionModel = pipe.unet
    vae = pipe.vae
    image_encoder: CLIPVisionModelWithProjection = pipe.image_encoder
    feature_extractor: CLIPImageProcessor = pipe.feature_extractor
    scheduler: EulerDiscreteScheduler = pipe.scheduler

    # ── Freeze everything, then add LoRA to UNet ───────────────────────────────
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = get_lora_config(rank=args.lora_rank, alpha=args.lora_rank)
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()  # Should be ~1-3% of total params

    unet.to(device, dtype=dtype)
    vae.to(device, dtype=dtype)
    image_encoder.to(device, dtype=dtype)

    # Gradient checkpointing — cuts VRAM usage significantly on long sequences
    unet.enable_gradient_checkpointing()

    # ── Dataset & DataLoader ───────────────────────────────────────────────────
    dataset = LandscapeClipDataset(args.data_dir, clip_len=args.clip_len, size=args.size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    # Linear warmup + cosine decay
    total_steps = len(dataloader) * args.epochs
    warmup_steps = total_steps // 10
    scheduler_lr = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
    )

    # ── Output dir ────────────────────────────────────────────────────────────
    run_dir = Path(args.output_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now():%H:%M:%S}] Saving checkpoints to {run_dir}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Training SVD LoRA | {len(dataset)} clips | {args.epochs} epochs")
    print(f"Batch size: {args.batch_size} | LR: {args.lr} | LoRA rank: {args.lora_rank}")
    print(f"{'='*55}\n")

    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        unet.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for frames, cond_frames in pbar:
            # frames      : (B, T, 3, H, W)  in [-1, 1]
            # cond_frames : (B, 3, H, W)      first frame

            frames = frames.to(device, dtype=dtype)
            cond_frames = cond_frames.to(device, dtype=dtype)

            B, T, C, H, W = frames.shape

            with torch.no_grad():
                # ── Encode frames to latent space via VAE ──────────────────
                # VAE expects (B*T, C, H, W)
                frames_flat = frames.reshape(B * T, C, H, W)
                latents = vae.encode(frames_flat).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.reshape(B, T, *latents.shape[1:])  # (B, T, 4, h, w)
                # SVD UNet expects (B, C, T, h, w)
                latents = latents.permute(0, 2, 1, 3, 4)

                # ── Encode conditioning image via CLIP ─────────────────────
                # feature_extractor expects PIL or numpy uint8 in [0, 255]
                cond_pil = (((cond_frames + 1) / 2).clamp(0, 1) * 255).byte()
                cond_pil = cond_pil.permute(0, 2, 3, 1).cpu().numpy()
                clip_inputs = feature_extractor(
                    images=[img for img in cond_pil],
                    return_tensors="pt",
                ).pixel_values.to(device, dtype=dtype)
                image_embeddings = image_encoder(clip_inputs).image_embeds  # (B, 1024)
                image_embeddings = image_embeddings.unsqueeze(1)             # (B, 1, 1024)

                # ── Add noise to latents (EDM scheduler — pas de add_noise standard) ──
                noise = torch.randn_like(latents)
                # Timesteps sous forme de float dans [0, 1] → sigma EDM
                sigmas = torch.rand(B, device=device, dtype=dtype)  # uniform noise level
                # Sigma EDM : sigma_max=700, sigma_min=0.002 (valeurs SVD)
                sigma_max, sigma_min = 700.0, 0.002
                sigmas_scaled = (sigma_max ** sigmas) * (sigma_min ** (1 - sigmas))
                sigmas_scaled = sigmas_scaled.view(B, 1, 1, 1, 1)
                noisy_latents = latents + sigmas_scaled * noise

                # Timesteps pour le UNet (log-sigma normalisé comme SVD l'attend)
                timesteps = (0.25 * torch.log(sigmas_scaled.squeeze())).to(device)
                if timesteps.dim() == 0:
                    timesteps = timesteps.unsqueeze(0)

                # ── Motion conditioning (fps_id, motion_bucket_id) ─────────
                # Using mid-range values — these will eventually be replaced by
                # your friend's flow signal
                added_time_ids = torch.tensor(
                    [[args.fps_id, args.motion_bucket_id, 0.0]] * B,
                    device=device, dtype=dtype
                )

            # ── Encode conditioning frame (première frame) ────────────────────
            with torch.no_grad():
                cond_latents = vae.encode(
                    cond_frames
                ).latent_dist.sample() * vae.config.scaling_factor
                # (B, 4, h, w) — pas d'expansion temporelle, SVD gère ça en interne


            with torch.no_grad():
                cond_latents = vae.encode(
                    cond_frames
                ).latent_dist.sample() * vae.config.scaling_factor  # (B, 4, h, w)
                # Répéter sur la dimension temporelle
                cond_latents_expanded = cond_latents.unsqueeze(2).expand(-1, -1, T, -1, -1)  # (B, 4, T, h, w)

            # Concat sur les canaux → (B, 8, T, h, w)
            unet_input = torch.cat([noisy_latents, cond_latents_expanded], dim=1)

            # SVD attend (B, T, 8, h, w) — il fait flatten(0,1) en interne
            unet_input_5d = unet_input.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, 8, h, w)

            pred_noise_flat = unet.base_model.model(
                unet_input_5d,
                timesteps,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
            ).sample  # (B*T, 4, h, w)

            pred_noise = pred_noise_flat.permute(0, 2, 1, 3, 4)

            # ── Loss: predict the noise (standard diffusion objective) ─────
            loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            scheduler_lr.step()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler_lr.get_last_lr()[0]:.2e}"})

        # ── End of epoch ───────────────────────────────────────────────────
        avg_loss = epoch_loss / len(dataloader)
        print(f"[{datetime.now():%H:%M:%S}] Epoch {epoch} | Avg loss: {avg_loss:.4f}")

        # Save latest LoRA weights
        unet.save_pretrained(str(run_dir / "lora_latest"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            unet.save_pretrained(str(run_dir / "lora_best"))
            print(f"  → New best saved (loss: {best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"LoRA weights saved to: {run_dir}")


# ─────────────────────────────────────────────
# Inference helper (quick test after training)
# ─────────────────────────────────────────────

def run_inference(lora_dir: str, image_path: str, output_path: str = "output.mp4"):
    """
    Quick test: load your fine-tuned LoRA and animate a single landscape image.
    """
    from diffusers import StableVideoDiffusionPipeline
    from peft import PeftModel
    import imageio

    dtype = torch.float16
    device = "cuda"

    print("Loading pipeline for inference...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=dtype,
        variant="fp16",
    )

    # Load LoRA weights
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
    pipe.to(device)
    pipe.vae = pipe.vae.to(dtype=torch.float16)  # SVD VAE has no fp16 variant; force cast after loading

    image = Image.open(image_path).convert("RGB").resize((512, 512))

    frames = pipe(
        image,
        num_frames=14,
        num_inference_steps=25,
        fps=7,
        motion_bucket_id=200,
        decode_chunk_size=4,   # decode 4 frames at a time to save VRAM
    ).frames[0]

    # Save as MP4
    imageio.mimsave(output_path, frames, fps=7)
    print(f"Saved to {output_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SVD temporal attention with LoRA")

    parser.add_argument("--data_dir",          type=str,   default="./data/svd_landscape")
    parser.add_argument("--output_dir",        type=str,   default="./checkpoints/svd_lora")
    parser.add_argument("--clip_len",          type=int,   default=14,     help="Frames per clip")
    parser.add_argument("--size",              type=int,   default=512,    help="Frame size")
    parser.add_argument("--epochs",            type=int,   default=3)
    parser.add_argument("--batch_size",        type=int,   default=1,      help="Keep at 1 for A100 safety")
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--lora_rank",         type=int,   default=16,     help="LoRA rank (8–32)")
    parser.add_argument("--fps_id",            type=int,   default=7,      help="SVD fps conditioning")
    parser.add_argument("--motion_bucket_id",  type=int,   default=127,    help="0=static, 255=lots of motion")

    # Inference mode
    parser.add_argument("--infer",             action="store_true")
    parser.add_argument("--lora_dir",          type=str,   default=None)
    parser.add_argument("--image_path",        type=str,   default=None)

    args = parser.parse_args()

    if args.infer:
        assert args.lora_dir and args.image_path, "Provide --lora_dir and --image_path for inference"
        run_inference(args.lora_dir, args.image_path)
    else:
        train(args)
