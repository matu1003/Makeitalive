import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LandscapeMotionDataset(Dataset):
    """
    PyTorch Dataset that loads the generated image pairs (I_t, I_{t+k}).
    """
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Path to the generated dataset folder (e.g. 'data/youtube_landscape'),
                      must contain the 'img_A' and 'img_B' sub-folders.
            transform: Function or composition of transforms (torchvision.transforms)
                       applied to the images.
        """
        self.data_dir = Path(data_dir)
        self.dir_A = self.data_dir / "img_A"
        self.dir_B = self.data_dir / "img_B"
        
        # Make sure both folders exist
        if not self.dir_A.exists() or not self.dir_B.exists():
            raise FileNotFoundError(f"Sub-folders 'img_A' and 'img_B' not found in {self.data_dir}")
            
        # List the files (assuming both folders use the same file names)
        # Sort them to guarantee a deterministic order
        self.image_filenames = sorted([f.name for f in self.dir_A.iterdir() if f.is_file() and f.suffix in ('.jpg', '.png')])
        
        if len(self.image_filenames) == 0:
            print(f"Warning: no image found in {self.dir_A}")
            
        # Default transform if none is given (converts the PIL image to a [C, H, W] tensor in [0, 1])
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
        
        # Full paths
        path_A = self.dir_A / filename
        path_B = self.dir_B / filename
        
        # Load the images with PIL (RGB)
        image_A = Image.open(path_A).convert('RGB')
        image_B = Image.open(path_B).convert('RGB')
        
        # Apply the transforms
        if self.transform:
            tensor_A = self.transform(image_A)
            tensor_B = self.transform(image_B)
            
        return tensor_A, tensor_B

# Basic usage example when the file is run directly
if __name__ == '__main__':
    dataset_path = "../../data/youtube_landscape"
    
    if os.path.exists(dataset_path):
        dataset = LandscapeMotionDataset(dataset_path)
        print(f"Dataset loaded with {len(dataset)} image pairs.")
        
        # Try loading the first pair
        if len(dataset) > 0:
            img_a, img_b = dataset[0]
            print(f"Tensor A shape: {img_a.shape}")
            print(f"Tensor B shape: {img_b.shape}")
    else:
        print(f"Cannot run the test, folder {dataset_path} does not exist yet.")
