import os
import argparse
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Ajout du dossier racine au PYTHONPATH pour permettre les imports absolus depuis 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import LandscapeMotionDataset
from src.motion_flow.model import MotionFlowUNet

def train(args):
    """
    Script d'entraînement du modèle Motion Flow.
    Conçu pour être lancé en arrière-plan (par exemple via tmux).
    """
    
    # 1. Configurer l'environnement (Device, Dossiers de logs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now()}] Lancement de l'entraînement sur: {device}")
    
    # Création du dossier pour sauvegarder les poids
    os.makedirs(args.ckpt_dir, exist_ok=True)
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"[{datetime.now()}] Les modèles seront sauvegardés dans {run_dir}\n")

    # 2. Préparer les données (Dataset & DataLoader)
    print(f"Chargement du dataset depuis {args.data_dir}...")
    try:
        dataset = LandscapeMotionDataset(args.data_dir)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"Dataset chargé: {len(dataset)} paires. ({len(dataloader)} batches par époque)")
    except Exception as e:
        print(f"Erreur lors du chargement du dataset: {e}")
        return

    # 3. Initialiser le Modèle, la Loss et l'Optimizer
    model = MotionFlowUNet().to(device)
    
    # La MSE (Mean Squared Error) est un bon point de départ pour pénaliser les erreurs de flow
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Boucle d'entraînement
    print("\n" + "="*50)
    print(f"Début de l'entraînement pour {args.epochs} époques")
    print(f"Taux d'apprentissage: {args.lr}")
    print("="*50 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Pbar pour la lisibilité si lancé en direct, mais tqdm gere bien stdout pour tmux
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch_idx, (img_A, img_B) in enumerate(pbar):
            # Transférer les images sur GPU (ou CPU)
            img_A = img_A.to(device)
            img_B = img_B.to(device)
            
            # --- FORWARD PASS ---
            # Objectif simpliste ici : on essaie de prédire un warp()
            # Dans une implémentation réaliste complète (ex: RAFT), on prédirait le flow 
            # et on comparerait le résultat warp(img_A, flow) avec img_B.
            # Pour l'instant, faisons au plus simple (on a besoin d'une baseline !).
            
            # Le réseau prédit le motion flow
            pred_flow = model(img_A)
            
            # TODO: Il nous manque la fonction de warping différentiable ! 
            # (torch.nn.functional.grid_sample).
            # En l'absence de "Ground Truth Flow" (vrai target en pixels),
            # nous ferons ici un simple mock-up pour vérifier que la pipeline tourne 
            # (Pénalisons le flow pour l'instant vers 0 juste pour compiler).
            
            loss = criterion(pred_flow, torch.zeros_like(pred_flow)) # /!\ LOSS TEMPORAIRE /!\
            
            # --- BACKWARD PASS ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update de la barre de progression
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Fin de l'époque
        avg_loss = epoch_loss / len(dataloader)
        print(f"[{datetime.now()}] Epoch {epoch}/{args.epochs} complétée. Loss Moyenne: {avg_loss:.4f}")
        
        # 5. Sauvegarde des poids (Checkpointing)
        # On sauvegarde le dernier
        torch.save(model.state_dict(), os.path.join(run_dir, "model_latest.pth"))
        
        # On sauvegarde le meilleur
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(run_dir, "model_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Nouveau meilleur modèle sauvegardé (Loss: {best_loss:.4f})")
            
    print("\nEntraînement terminé !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle de Motion Flow")
    
    # Paramètres des Données
    parser.add_argument("--data_dir", type=str, default="./data/youtube_landscape", help="Dossier du dataset")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Dossier pour sauvegarder les modèles")
    
    # Hyperparamètres d'entraînement
    parser.add_argument("--epochs", type=int, default=50, help="Nombre total d'époques")
    parser.add_argument("--batch_size", type=int, default=8, help="Taille des batchs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate de l'optimizer")
    parser.add_argument("--num_workers", type=int, default=4, help="Nombre de processus pour le dataloader")
    
    args = parser.parse_args()
    train(args)
