import torch
import torch.nn.functional as F


def warp(img, flow):
    """
    Backward-warps a batch of images with a dense flow field (in pixels):
    output(x) = img(x + flow(x)), bilinear sampling with border padding.

    Args:
        img:  (B, C, H, W) tensor.
        flow: (B, 2, H, W) tensor, channel 0 = dx, channel 1 = dy.
    """
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    
    flow_normalized = torch.stack([
        flow[:, 0] / (W / 2),
        flow[:, 1] / (H / 2)
    ], dim=-1)
    
    grid = grid + flow_normalized
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
