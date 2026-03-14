import os
import cv2
import yt_dlp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def histogram_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """Distance entre deux images basée sur leur histogramme de couleurs."""
    #seuil : 0.3–0.4
    dist = 0
    for channel in range(3):  # B, G, R
        hist1 = cv2.calcHist([img1], [channel], None, [64], [0, 256])
        hist2 = cv2.calcHist([img2], [channel], None, [64], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        dist += cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return dist / 3  # Moyenne sur les 3 canaux, valeur entre 0 et 1

def mse_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcule la distance MSE (Mean Squared Error) entre deux images.
    Une valeur plus basse indique une similarité plus grande.
    
    Args:
        img1: Première image (numpy array).
        img2: Deuxième image (numpy array).
    
    Returns:
        La distance MSE.
    """
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32))**2)

def scene_change_distance(img1: np.ndarray, img2: np.ndarray, 
                           thumb_size: int = 16) -> float:
    """
    Compare deux images réduites en niveaux de gris.
    Insensible aux petits mouvements, sensible aux changements de scène.
    """
    # Seuil : 0.15–0.25
    t1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (thumb_size, thumb_size))
    t2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (thumb_size, thumb_size))
    return np.mean(np.abs(t1.astype(np.float32) - t2.astype(np.float32))) / 255.0


def is_scene_change(img1: np.ndarray, img2: np.ndarray,
                    hist_threshold: float = 0.30,
                    thumb_threshold: float = 0.20) -> bool:
    return (histogram_distance(img1, img2) > hist_threshold or 
            scene_change_distance(img1, img2) > thumb_threshold)

# def image_distance(img1: np.ndarray, img2: np.ndarray) -> float:

def download_and_extract_pairs(
    youtube_url: str, 
    output_dir: str, 
    sample_every_n_seconds: float = 5.0,
    frame_gap: int = 3,
    target_size: int = 512,
    max_pairs: int = -1,
    clean: float = False,
):
    """
    Télécharge un flux vidéo YouTube (en mémoire ou via un fichier temporaire) 
    et extrait des paires d'images (I_t, I_{t+k}) pour l'entraînement au motion flow.
    
    Args:
        youtube_url: l'URL de la vidéo.
        output_dir: le dossier où enregistrer les données.
        sample_every_n_seconds: l'intervalle de temps entre deux échantillons (en secondes).
        frame_gap: l'écart en nombre de frames entre la frame A et la frame B.
        target_size: la taille (carrée) des images extraites (ex: 512).
        max_pairs: le nombre maximum de paires à extraire. -1 pour tout extraire.
        clean: seuil de distance MSE pour filtrer les paires. Si None, pas de filtrage.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Dossiers pour la frame de départ (img_A) et la frame d'arrivée (img_B)
    dir_A = out_path / "img_A"
    dir_B = out_path / "img_B"
    dir_A.mkdir(exist_ok=True)
    dir_B.mkdir(exist_ok=True)

    print(f"Extraction des URL de stream avec yt-dlp pour {youtube_url}...")
    
    # Configuration de yt-dlp pour récupérer l'URL du flux direct (évite le téléchargement complet)
    ydl_opts = {
        'format': 'best[ext=mp4]/best', # On préfère le mp4 pour la compatibilité OpenCV
        'quiet': True,
        'noplaylist': True,
    }
    
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            stream_url = info['url']
            fps_video = info.get('fps', 30)
            if fps_video is None:
                fps_video = 30
    except Exception as e:
        print(f"Erreur lors de l'extraction yt-dlp: {e}")
        return

    print(f"Ouverture du flux vidéo (FPS estimé: {fps_video})...")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir le flux vidéo.")
        return

    # Calcul des intervalles en nombre de frames
    frame_interval = int(fps_video * sample_every_n_seconds)
    
    # On utilise tqdm pour afficher la progression
    pbar = tqdm(desc="Paires extraites", unit="paires")
    
    pair_count = 0
    current_frame_idx = 0
    deleted_pairs = 0
    
    while True:
        # Lire la frame courante (Frame A)
        ret_A, frame_A = cap.read()
        if not ret_A:
            break
            
        # Si c'est une frame qui nous intéresse selon notre intervalle
        if current_frame_idx % frame_interval == 0:
            # On cherche la frame B (Frame A + frame_gap)
            frame_B = None
            
            # Avancer le curseur jusqu'à la frame B
            for _ in range(frame_gap):
                ret_B, temp_B = cap.read()
                current_frame_idx += 1
                if not ret_B:
                    break
                frame_B = temp_B
                
            if frame_B is None:
                break # Fin de la vidéo atteinte pendant le gap
                
            # Traitement des deux frames
            # 1. Redimensionnement (Resize et Center Crop pour faire un carré parfait)
            img_A = preprocess_frame(frame_A, target_size)
            img_B = preprocess_frame(frame_B, target_size)
            
            # 2. Filtrage basé sur la distance si seuil défini
            if clean:
                if is_scene_change(img_A, img_B):
                    # Paires trop différentes, on passe
                    deleted_pairs += 1
                    continue
            
            # Sauvegarde des images sous forme de JPEG haute qualité
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
    print(f"Extraction terminée ! {pair_count} paires sauvegardées dans {output_dir}. {deleted_pairs} paires filtrées.")

def preprocess_frame(frame: np.ndarray, target_size: int) -> np.ndarray:
    """
    Redimensionne l'image pour que le plus petit côté soit égal à target_size,
    puis fait un recadrage au centre (Center Crop) pour obtenir un carré parfait.
    """
    h, w = frame.shape[:2]
    
    # Resize proportionnel
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
        
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center Crop
    start_y = (new_h - target_size) // 2
    start_x = (new_w - target_size) // 2
    
    cropped = resized[start_y:start_y+target_size, start_x:start_x+target_size]
    
    return cropped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraire un dataset de paires depuis YouTube")
    parser.add_argument("--url", type=str, required=True, help="URL de la vidéo YouTube")
    parser.add_argument("--name", type=str, default="youtube_landscape", help="Nom du dataset (créera un dossier ./data/NOM)")
    parser.add_argument("--interval", type=float, default=5.0, help="Secondes entre chaque paire extraite")
    parser.add_argument("--gap", type=int, default=3, help="Écart de frames (ex: 3ème frame après A)")
    parser.add_argument("--size", type=int, default=512, help="Taille des images en sortie (carré)")
    parser.add_argument("--max_pairs", type=int, default=-1, help="Nombre max de paires (pour tester)")
    parser.add_argument("--clean", action="store_true", help="Activer le filtrage des paires distantes")    
    args = parser.parse_args()
    
    out_dir = os.path.join(".", "data", args.name)
    print(f"Paramètres : URL={args.url}, Output={out_dir}, Interval={args.interval}s, Gap={args.gap} frames, Size={args.size}px, Max Pairs={args.max_pairs}, Clean Threshold={args.clean}")
    download_and_extract_pairs(
        youtube_url=args.url,
        output_dir=out_dir,
        sample_every_n_seconds=args.interval,
        frame_gap=args.gap,
        target_size=args.size,
        max_pairs=args.max_pairs,
        clean=args.clean,
    )
