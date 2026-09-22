import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from data.dataset import LandscapeMotionDataset
from motion_flow.model import MotionFlowUNet
from motion_flow.warp import warp

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Mixed precision is only used on GPU
    use_amp = device.type == 'cuda'
    print(f"[{datetime.now()}] Starting training on: {device} (AMP: {use_amp})")
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[{datetime.now()}] Models will be saved to {run_dir}\n")

    print(f"Loading dataset from {args.data_dir}...")
    try:
        dataset = LandscapeMotionDataset(args.data_dir)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=use_amp
        )
        print(f"Dataset loaded: {len(dataset)} pairs ({len(dataloader)} batches per epoch)")
    except Exception as e:
        print(f"Error while loading the dataset: {e}")
        return

    model = MotionFlowUNet().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("\n" + "="*50)
    print(f"Training for {args.epochs} epochs")
    print(f"Learning rate: {args.lr}")
    print("="*50 + "\n")
    
    best_loss = float('inf')
    scaler = GradScaler(device=device.type, enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch_idx, (img_A, img_B) in enumerate(pbar):
            img_A = img_A.to(device)
            img_B = img_B.to(device)
            
            # --- FORWARD PASS ---
            with autocast(device_type=device.type, enabled=use_amp):
                pred_flow = model(img_A)
                img_B_pred = warp(img_A, pred_flow)
                loss = criterion(img_B_pred, img_B)

            # --- BACKWARD PASS ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"[{datetime.now()}] Epoch {epoch}/{args.epochs} done. Mean loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(run_dir, "model_latest.pth"))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(run_dir, "model_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (loss: {best_loss:.4f})")
            
    print("\nTraining complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Motion Flow model")
    
    parser.add_argument("--data_dir", type=str, default="./data/youtube_landscape", help="Dataset folder")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Folder where the models are saved")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader worker processes")
    
    args = parser.parse_args()
    train(args)