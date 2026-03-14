import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LandscapeMotionDataset(Dataset):
    """
    PyTorch Dataset pour charger les paires d'images (I_t, I_{t+k}) générées.
    """
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Chemin vers le dossier généré (ex: 'data/youtube_landscape')
                      doit contenir les sous-dossiers 'img_A' et 'img_B'.
            transform: Fonction ou composition de transformations (torchvision.transforms)
                       à appliquer aux images.
        """
        self.data_dir = Path(data_dir)
        self.dir_A = self.data_dir / "img_A"
        self.dir_B = self.data_dir / "img_B"
        
        # On vérifie que les dossiers existent
        if not self.dir_A.exists() or not self.dir_B.exists():
            raise FileNotFoundError(f"Les sous-dossiers 'img_A' et 'img_B' sont introuvables dans {self.data_dir}")
            
        # On liste les fichiers (en supposant qu'ils aient été générés avec le même nom dans les deux dossiers)
        # On trie pour garantir l'ordre
        self.image_filenames = sorted([f.name for f in self.dir_A.iterdir() if f.is_file() and f.suffix in ('.jpg', '.png')])
        
        if len(self.image_filenames) == 0:
            print(f"Attention: Aucune image trouvée dans {self.dir_A}")
            
        # Transformations par défaut si aucune n'est fournie (Convertit l'image PIL en Tensor [C, H, W] entre 0 et 1)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.image_filenames[idx]
        
        # Chemins complets
        path_A = self.dir_A / filename
        path_B = self.dir_B / filename
        
        # Loader les images avec PIL (RGB)
        image_A = Image.open(path_A).convert('RGB')
        image_B = Image.open(path_B).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            tensor_A = self.transform(image_A)
            tensor_B = self.transform(image_B)
            
        return tensor_A, tensor_B

# Exemple d'utilisation basique si le fichier est exécuté
if __name__ == '__main__':
    dataset_path = "../../data/youtube_landscape"
    
    if os.path.exists(dataset_path):
        dataset = LandscapeMotionDataset(dataset_path)
        print(f"Dataset chargé avec {len(dataset)} paires d'images.")
        
        # Tester le chargement de la première paire
        if len(dataset) > 0:
            img_a, img_b = dataset[0]
            print(f"Shape du tenseur A: {img_a.shape}")
            print(f"Shape du tenseur B: {img_b.shape}")
    else:
        print(f"Impossible de tester, le dossier {dataset_path} n'existe pas encore.")
