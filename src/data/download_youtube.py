import os
import argparse
import yt_dlp

def download_video(url: str, output_path: str, max_height: int = 720):
    """
    Télécharge une vidéo YouTube (uniquement la vidéo, pas l'audio) en limitant la résolution 
    pour éviter les fichiers gigantesques.
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # On force le codec vidéo à H.264 (avc1) pour s'assurer que OpenCV pourra toujours le lire.
    # On précise qu'on veut le meilleur format mp4 dont la hauteur est <= max_height
    # Et on ne télécharge pas l'audio.
    ydl_opts = {
        'format': f'bestvideo[vcodec^=avc1][height<={max_height}][ext=mp4]/best[vcodec^=avc1][height<={max_height}][ext=mp4]/bestvideo[height<={max_height}][ext=mp4]',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
    }
    
    print(f"Téléchargement de la vidéo depuis {url} (Résolution max: {max_height}p)...")
    print(f"Fichier de destination : {output_path}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("\\nTéléchargement terminé avec succès !")
    except Exception as e:
        print(f"\\nErreur lors du téléchargement : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharger une vidéo YouTube localement")
    parser.add_argument("--url", type=str, required=True, help="URL de la vidéo YouTube")
    parser.add_argument("--out", type=str, default="./data/source_video.mp4", help="Chemin du fichier de sortie (ex: ./data/video.mp4)")
    parser.add_argument("--height", type=int, default=720, help="Résolution maximale (ex: 720 pour 720p, 480 pour 480p)")
    
    args = parser.parse_args()
    
    download_video(args.url, args.out, max_height=args.height)
