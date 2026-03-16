import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def tensor_to_img255(t):  # [-1,1] -> [0,255] uint8
    x = (t.clamp(-1,1)*0.5+0.5).mul(255).byte().permute(1,2,0).cpu().numpy()
    return x

def measure_psnr_ssim(fake, gt):
    f = tensor_to_img255(fake); g = tensor_to_img255(gt)
    v_psnr = psnr(g, f, data_range=255)
    v_ssim = ssim(g, f, channel_axis=2, data_range=255)
    return v_psnr, v_ssim

def try_niqe(img_numpy_uint8):
    try:
        import piq, torch
        x = torch.from_numpy(img_numpy_uint8).float().permute(2,0,1).unsqueeze(0)/255.0
        return float(piq.niqe(x))
    except Exception:
        return None
