import math
import torch
import json
import os
import numpy as np
import imageio
from tqdm import tqdm
from submodules.lpipsPyTorch import LPIPSEval
from fused_ssim import fused_ssim
from utils.train_util import render
import matplotlib.pyplot as plt

VERT_BUDGET = 500000
TIME_BUDGET = 600

torch.set_printoptions(precision=10)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def evaluate(model, splits, output_path, tile_size, min_t, save=True):
    gt_path = os.path.join(output_path, "images", "gt")
    pred_path = os.path.join(output_path, "images", "pred")
    if save:
        os.makedirs(gt_path, exist_ok=True)
        os.makedirs(pred_path, exist_ok=True)
    
    # Initialize LPIPS
    lpips_eval = LPIPSEval(net_type='vgg', device='cuda')
    
    short_results = {}
    results = {}
    for split, cameras in splits:
        renders, gts = [], []
        ssims, psnrs, lpipss = [], [], []
        if save:
            os.makedirs(os.path.join(gt_path, split), exist_ok=True)
            os.makedirs(os.path.join(pred_path, split), exist_ok=True)
        
        for idx, camera in enumerate(tqdm(cameras, desc=f"Rendering {split} set")):
            with torch.no_grad():
                with torch.no_grad():
                    render_pkg = render(camera, model, tile_size=tile_size, min_t=min_t)
                image = render_pkg['render'].clip(min=0, max=1).unsqueeze(0)
                # image = image.permute(1, 2, 0).detach()
                
                # Load corresponding ground truth image
                gt = camera.original_image.cuda().unsqueeze(0)
                
                # Compute metrics
                ssim_val = fused_ssim(image, gt).item()
                psnr_val = psnr(image, gt).item()
                lpips_val = lpips_eval.criterion(2 * image - 1, 2 * gt - 1).item()
                
                # Store results
                renders.append(image)
                gts.append(gt)
                ssims.append(ssim_val)
                psnrs.append(psnr_val)
                lpipss.append(lpips_val)
                
                # Save individual images
                if save:
                    imageio.imwrite(os.path.join(pred_path, split, f"{idx:04d}.png"), (image.cpu()[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    imageio.imwrite(os.path.join(gt_path, split, f"{idx:04d}.png"), (gt.cpu()[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                
                # Save per-image metrics
                results[f"{split}_{idx:04d}"] = {"SSIM": ssim_val, "PSNR": psnr_val, "LPIPS": lpips_val}
        
        means = {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item()
        }
        # Compute mean metrics
        results[f"{split}_mean"] = means
        short_results = {**short_results,
            f"{split}_SSIM": torch.tensor(ssims).mean().item(),
            f"{split}_PSNR": torch.tensor(psnrs).mean().item(),
            f"{split}_LPIPS": torch.tensor(lpipss).mean().item()
        }
        
    return short_results
