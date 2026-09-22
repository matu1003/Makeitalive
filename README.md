# Makeitalive: Landscape Picture Animation

<p align="center">
  <img src="assets/gifs/svd_final_1.gif" alt="Landscape animated by our LoRA fine-tuned Stable Video Diffusion" width="100%">
  <br>
  <em>A single static landscape photo brought to life by our LoRA fine-tuned Stable Video Diffusion model.</em>
</p>

<p align="center">
    <a href="docs/report.pdf"><b>Report</b></a> &nbsp;|&nbsp;
    <a href="docs/poster_1.pdf"><b>Poster 1</b></a> &nbsp;|&nbsp;
    <a href="docs/poster_2.pdf"><b>Poster 2</b></a>
</p>

## Project Overview

**Makeitalive** is a computer vision and generative AI project developed by Arthur Fournier and Mathurin Petit for the `CSC_52002 - Generative AI Project` course at Ecole Polytechnique. The goal of this project is to breathe life into static landscape photographs by animating them into realistic video sequences. We explored and implemented two distinct methodologies:

1. **Motion Flow (Warping):** A lightweight, self-supervised U-Net architecture that predicts a dense `(dx, dy)` optical flow field from a single image. The model displaces existing pixels to synthesize motion. It is exceptionally fast but lacks the ability to generate new visual details for occluded areas and is not suitable for realistic animation.
2. **Stable Video Diffusion + LoRA fine-tuning:** A generative approach built on Stability AI's `stable-video-diffusion-img2vid`, whose temporal attention layers are fine-tuned with **LoRA** (`diffusers` + `peft`) on a curated dataset of drone landscape footage. This heavy-weight approach successfully hallucinates rich textures and sweeping, cinematic parallax motion from a single static input.

---

## Results

### 1. Naive approach: Motion Flow warping

The U-Net predicts a flow field that is applied iteratively to the input image. Motion is plausible for textures like wheat or water, but no new content can be created.

| Wheat | Lake |
|:---:|:---:|
| <img src="assets/gifs/motion_flow_wheat.gif" width="320"> | <img src="assets/gifs/motion_flow_lake.gif" width="320"> |

### 2. Stable Video Diffusion (SVD), out of the box, without LoRA

The pretrained SVD model without any fine-tuning: the motion is short and does not produce the drone-like camera movement we are looking for.

<p align="center">
  <img src="assets/gifs/svd_not_trained.gif" width="480">
</p>

### 3. SVD + LoRA, fine-tuned on drone landscapes

After LoRA fine-tuning of its temporal attention layers on landscape drone clips, the model produces smooth camera motion with convincing parallax and generates the newly revealed parts of the scene.

| | |
|:---:|:---:|
| <img src="assets/gifs/svd_final_2.gif" width="320"> | <img src="assets/gifs/svd_final_3.gif" width="320"> |

<p align="center">
  <img src="assets/gifs/svd_final_1.gif" width="640">
</p>

---

## Project Architecture

```plaintext
Makeitalive/
├── assets/
│   ├── gifs/                   # Animated results shown in this README
│   ├── images/                 # Sample input picture (landscape.jpg)
│   └── videos/                 # Full-quality result videos (motion_flow/, svd/)
├── docs/                       # Final report (CVPR format) and the two project posters
├── notebooks/                  # Guided walkthroughs built on top of src/ (see below)
└── src/
    ├── data/                   # YouTube download, image-pair extraction, scene-change filtering, PyTorch dataset
    ├── motion_flow/            # U-Net model, warping, self-supervised training and inference
    └── svd_lora/               # SVD clip extraction, LoRA fine-tuning and inference
```

### Notebooks

All the logic lives in `src/`; the notebooks walk through each step with visualizations and run on Colab or locally.

| Notebook | Content | |
|---|---|:---:|
| [`01_dataset`](notebooks/01_dataset.ipynb) | Download the drone footage, extract image pairs, scene-change filtering, optical flow of a pair | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matu1003/Makeitalive/blob/main/notebooks/01_dataset.ipynb) |
| [`02_motion_flow`](notebooks/02_motion_flow.ipynb) | Warping intuition, U-Net, self-supervised training, predicted flow and animations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matu1003/Makeitalive/blob/main/notebooks/02_motion_flow.ipynb) |
| [`03_svd_lora`](notebooks/03_svd_lora.ipynb) | Clip extraction, LoRA fine-tuning of SVD, pretrained vs fine-tuned comparison (GPU runtime required) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matu1003/Makeitalive/blob/main/notebooks/03_svd_lora.ipynb) |

---

## Installation

The project uses [uv](https://docs.astral.sh/uv/).
```bash
uv sync                 # Motion Flow + data tooling
uv sync --extra svd     # + diffusers, transformers, accelerate, peft for SVD
```
Downloaded videos, datasets, checkpoints and generated animations are stored locally in `data/`, `checkpoints/` and `outputs/` (git-ignored).

---

## Important Commands

### 1. Download YouTube Drone Landscapes
Download a large compilation drone video (e.g., *10 Hours Fantastic Views of Nature 4K*) to build your dataset.
```bash
uv run src/data/download_youtube.py \
    --url "https://www.youtube.com/watch?v=AKeUssuu3Is" \
    --out "./data/10hourslandscape.mp4"
```

### 2. Make Dataset (From Local Video)
Extract consecutive, motion-filtered image pairs from a downloaded video with strict thresholds to drop montage cuts.
```bash
uv run src/data/make_dataset_video.py \
    --video "./data/10hourslandscape.mp4" \
    --name "dataset_local" \
    --interval 5.0 \
    --gap 4 \
    --size 512 \
    --max_pairs 100 \
    --clean
```

### 3. Make Dataset (Direct from YouTube)
Stream and extract the training pairs directly without downloading the full video first.
```bash
uv run src/data/make_dataset_youtube.py \
    --url "https://www.youtube.com/watch?v=AKeUssuu3Is" \
    --name "ytb_gap3_interval5dot0_clean" \
    --interval 5.0 \
    --gap 3 \
    --size 512 \
    --max_pairs 5000 \
    --clean
```

### 4. Training the Motion Flow Model
Start self-supervised training of the MotionFlow U-Net using the extracted pairs.
```bash
uv run src/motion_flow/train.py \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --num_workers 16
```

### 5. Animate a Picture with Motion Flow
Predict the flow of a picture and animate it (uses the latest checkpoint in `./checkpoints` if `--checkpoint` is omitted).
```bash
uv run src/motion_flow/infer.py \
    --image "./assets/images/landscape.jpg" \
    --out "./outputs/landscape.gif" \
    --magnitude 25 \
    --ping_pong
```
Add `--autoregressive` to re-predict the flow on every frame.

### 6. Make the SVD + LoRA Clip Dataset
Extract short, motion-filtered 14-frame clips (scene cuts removed) for SVD fine-tuning.
```bash
uv run src/svd_lora/make_dataset_svd.py \
    --url "https://www.youtube.com/watch?v=AKeUssuu3Is" \
    --out "./data/svd_landscape" \
    --clip_len 14 \
    --fps 7 \
    --size 512 \
    --max_clips 5000
```

### 7. Fine-tune SVD with LoRA and Run Inference
Fine-tune the temporal attention layers of SVD (an A100-class GPU is recommended; the SVD weights are downloaded from Hugging Face).
```bash
uv run src/svd_lora/train_svd_lora.py \
    --data_dir "./data/svd_landscape" \
    --output_dir "./checkpoints/svd_lora" \
    --epochs 3 \
    --batch_size 1 \
    --lora_rank 16
```
Animate a picture with a trained LoRA (omit `--lora_dir` to use the pretrained SVD as a baseline):
```bash
uv run src/svd_lora/train_svd_lora.py --infer \
    --lora_dir "./checkpoints/svd_lora/run_<timestamp>/lora_best" \
    --image_path "./assets/images/landscape.jpg" \
    --out "./outputs/svd_lora.mp4"
```

---

## Authors

- **Arthur Fournier**
- **Mathurin Petit**
