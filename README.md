# Makeitalive
Multimodal generation to animate pictures

Overleaf report link : https://overleaf.enst.fr/read/mdrkrjfvdvfh#c43d55

# Datasets
https://huggingface.co/datasets/Loie/VGGSound

### Using a Youtube Video
https://www.youtube.com/watch?v=AKeUssuu3Is&list=RDAKeUssuu3Is&start_radio=1&t=10196s

Lancer la collecte du dataset
```bash
uv run src/data/make_dataset_youtube.py \
    --url "https://www.youtube.com/watch?v=AKeUssuu3Is" \
    --out "./data/youtube_landscape" \
    --interval 5.0 \
    --gap 3 \
    --size 512 \
    --max_pairs 100
```

Lancer l'entraînement
```bash
uv run src/motion_flow/train.py --epochs 100 --batch_size 8 --lr 1e-4 --num_workers 16
```