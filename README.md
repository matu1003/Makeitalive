# Makeitalive: Landscape Picture Animation

**Overleaf report link:** [https://overleaf.enst.fr/read/mdrkrjfvdvfh#c43d55](https://overleaf.enst.fr/read/mdrkrjfvdvfh#c43d55)

## Project Overview

**Makeitalive** is a computer vision and generative AI project developed by Arthur Fournier and Mathurin Petit for the `CSC_52002 - Generative AI Project` course. The goal of this project is to breathe life into static landscape photographs by animating them into realistic video sequences. We explored and implemented two distinct methodologies:

1. **Motion Flow (Warping):** A lightweight, self-supervised U-Net architecture that predicts a dense `(dx, dy)` optical flow field from a single image. The model displaces existing pixels to synthesize motion. It is exceptionally fast but lacks the ability to generate new visual details for occluded areas.
2. **Stable Video Diffusion (SVD):** A generative AI approach fine-tuning Stability AI's SVD model (`SVD_Xtend`) on a curated dataset of drone landscape footage. This heavy-weight approach successfully hallucinates rich textures and sweeping, cinematic parallax motion from a single static input.

---

## Project Architecture

```plaintext
Makeitalive/
├── data/                       # Local directory to store downloaded videos and extracted image pairs
├── notebooks/                  # Jupyter notebooks for data exploration, model prototyping, and inference
├── poster/                     # LaTeX files and compiled PDFs for the project poster presentations
│   └── images/                 # All figures and visual assets used in the posters
├── report/                     # Master IEEE CVPR structured academic report (fournier_petit.pdf)
│   └── images/                 # All figures and visual assets used in the report
└── src/                        # Main source code directory
    ├── data/                   # Scripts for downloading YouTube videos and extracting/filtering datasets
    │   ├── download_youtube.py
    │   ├── make_dataset_video.py
    │   └── make_dataset_youtube.py
    └── motion_flow/            # Motion Flow U-Net model definitions and training routines
        └── train.py
```

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