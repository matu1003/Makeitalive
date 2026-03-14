import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def histogram_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """Distance entre deux images basée sur leur histogramme de couleurs."""
    dist = 0
    for channel in range(3):  # B, G, R
        hist1 = cv2.calcHist([img1], [channel], None, [64], [0, 256])
        hist2 = cv2.calcHist([img2], [channel], None, [64], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        dist += cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return dist / 3  # Moyenne sur les 3 canaux, valeur entre 0 et 1

def scene_change_distance(img1: np.ndarray, img2: np.ndarray, 
                           thumb_size: int = 16) -> float:
    """
    Compare deux images réduites en niveaux de gris.
    Insensible aux petits mouvements, sensible aux changements de scène.
    """
    t1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (thumb_size, thumb_size))
    t2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (thumb_size, thumb_size))
    return np.mean(np.abs(t1.astype(np.float32) - t2.astype(np.float32))) / 255.0

def is_scene_change(img1: np.ndarray, img2: np.ndarray,
                    hist_threshold: float = 0.30,
                    thumb_threshold: float = 0.20) -> bool:
    return (histogram_distance(img1, img2) > hist_threshold or 
            scene_change_distance(img1, img2) > thumb_threshold)

def extract_pairs_from_video(
    video_path: str, 
    output_dir: str, 
    sample_every_n_seconds: float = 5.0,
    frame_gap: int = 3,
    target_size: int = 512,
    max_pairs: int = -1,
    clean: bool = False,
):
    """
    Lit un fichier vidéo local et extrait des paires d'images (I_t, I_{t+k}) 
    pour l'entraînement.
    """
    if not os.path.exists(video_path):
        print(f"Erreur: Le fichier {video_path} n'existe pas.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    dir_A = out_path / "img_A"
    dir_B = out_path / "img_B"
    dir_A.mkdir(exist_ok=True)
    dir_B.mkdir(exist_ok=True)

    print(f"Ouverture du fichier local : {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir le fichier vidéo.")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30
    print(f"FPS détecté : {fps_video:.2f} images/seconde")

    frame_interval = int(fps_video * sample_every_n_seconds)
    
    pbar = tqdm(desc="Paires extraites", unit="paires")
    
    pair_count = 0
    current_frame_idx = 0
    deleted_pairs = 0
    
    while True:
        ret_A, frame_A = cap.read()
        if not ret_A:
            break
            
        if current_frame_idx % frame_interval == 0:
            frame_B = None
            for _ in range(frame_gap):
                ret_B, temp_B = cap.read()
                current_frame_idx += 1
                if not ret_B:
                    break
                frame_B = temp_B
                
            if frame_B is None:
                break
                
            img_A = preprocess_frame(frame_A, target_size)
            img_B = preprocess_frame(frame_B, target_size)
            
            if clean:
                if is_scene_change(img_A, img_B):
                    deleted_pairs += 1
                    continue
            
            filename = f"pair_{pair_count:06d}.jpg"
            cv2.imwrite(str(dir_A / filename), img_A)
            cv2.imwrite(str(dir_B / filename), img_B)
            
            pair_count += 1
            pbar.update(1)
            
            if max_pairs > 0 and pair_count >= max_pairs:
                print(f"\\nLimite de {max_pairs} paires atteinte.")
                break
        else:
            current_frame_idx += 1
            
    cap.release()
    pbar.close()
    print(f"Extraction terminée ! {pair_count} paires sauvegardées dans {output_dir}.")
    if clean:
        print(f"{deleted_pairs} paires filtrées (changements de plan détectés).")

def preprocess_frame(frame: np.ndarray, target_size: int) -> np.ndarray:
    """
    Redimensionne l'image pour que le plus petit côté soit égal à target_size,
    puis fait un recadrage au centre (Center Crop) pour obtenir un carré parfait.
    """
    h, w = frame.shape[:2]
    
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
        
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    start_y = (new_h - target_size) // 2
    start_x = (new_w - target_size) // 2
    
    cropped = resized[start_y:start_y+target_size, start_x:start_x+target_size]
    
    return cropped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraire un dataset de paires depuis une vidéo locale")
    parser.add_argument("--video", type=str, required=True, help="Chemin vers le fichier vidéo local (.mp4, .mkv, etc.)")
    parser.add_argument("--name", type=str, default="local_dataset", help="Nom du dataset (créera un dossier ./data/NOM)")
    parser.add_argument("--interval", type=float, default=5.0, help="Secondes entre chaque paire extraite")
    parser.add_argument("--gap", type=int, default=3, help="Écart de frames (ex: 3ème frame après A)")
    parser.add_argument("--size", type=int, default=512, help="Taille des images en sortie (carré)")
    parser.add_argument("--max_pairs", type=int, default=-1, help="Nombre max de paires (-1 = toutes)")
    parser.add_argument("--clean", action="store_true", help="Activer le filtrage des paires avec un changement de plan")    
    args = parser.parse_args()
    
    out_dir = os.path.join(".", "data", args.name)
    print(f"Paramètres : Video={args.video}, Output={out_dir}, Interval={args.interval}s, Gap={args.gap} frames, Size={args.size}px")
    
    extract_pairs_from_video(
        video_path=args.video,
        output_dir=out_dir,
        sample_every_n_seconds=args.interval,
        frame_gap=args.gap,
        target_size=args.size,
        max_pairs=args.max_pairs,
        clean=args.clean,
    )
