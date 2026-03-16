import os
import yaml
import time
import csv
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils

from models.discriminator_patch import DiscriminatorLite
from data.dataset import PairDataset
from losses import GANLoss, PerceptualLoss, FeatureConsistency, EdgeLoss
from utils.metrics import measure_psnr_ssim, tensor_to_img255, try_niqe


# =========================
# 选择生成器结构
# =========================
def get_generator(cfg):
    gen_type = cfg.get("generator_type", "mobile").lower()
    if gen_type == "funie":
        from models.generator_funie import GeneratorFUNIE
        print("[INFO] Using baseline generator: FUnIE-GAN")
        return GeneratorFUNIE()
    else:
        from models.generator_mobile import GeneratorMobile
        print("[INFO] Using improved generator: MobileGAN")
        return GeneratorMobile()


# =========================
# 工具函数
# =========================
@torch.no_grad()
def denorm(x):
    """
    仅用于保存图片：
    将 [-1, 1] 恢复到 [0, 1]
    """
    return x.clamp(-1, 1) * 0.5 + 0.5


def save_image_grid(x, path, nrow=4):
    vutils.save_image(denorm(x), path, nrow=nrow)


def set_seed(seed: int = 42):
    """
    固定随机种子，提升可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_stage_weights(epoch, total_epochs, cfg):
    """
    分阶段动态权重策略（期刊版新增）

    训练分三阶段：
    1) 前期：重建优先（L1、LPIPS 更大）
    2) 中期：平衡阶段
    3) 后期：增强真实感、语义稳定和边缘细节

    你后续如果实验效果一般，可以只在这里调权重，
    不需要改主训练逻辑。
    """
    # 基础权重（来自 config）
    base_adv = cfg['lambda_adv']
    base_l1 = cfg['lambda_l1']
    base_lpips = cfg['lambda_lpips']
    base_feat = cfg['lambda_feat']
    base_edge = cfg.get('lambda_edge', 0.1)

    r = epoch / total_epochs

    # 阶段1：前 30%
    if r <= 0.30:
        return {
            'adv':   base_adv * 0.7,
            'l1':    base_l1 * 1.2,
            'lpips': base_lpips * 1.1,
            'feat':  base_feat * 0.8,
            'edge':  base_edge * 0.8,
        }

    # 阶段2：中间 40%
    elif r <= 0.70:
        return {
            'adv':   base_adv,
            'l1':    base_l1,
            'lpips': base_lpips,
            'feat':  base_feat,
            'edge':  base_edge,
        }

    # 阶段3：后 30%
    else:
        return {
            'adv':   base_adv * 1.2,
            'l1':    base_l1 * 0.9,
            'lpips': base_lpips,
            'feat':  base_feat * 1.2,
            'edge':  base_edge * 1.3,
        }


def main(cfg_path='config.yaml', device='cuda'):
    # ========== 读取配置 ==========
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))

    # 随机种子
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # 输出目录
    for d in ['checkpoints', 'samples', 'metrics', 'logs']:
        os.makedirs(os.path.join(cfg['out_dir'], d), exist_ok=True)

    # ========== 数据 ==========
    ds_tr = PairDataset(cfg['train_lr'], cfg['train_hr'], cfg['img_size'])
    ds_va = PairDataset(cfg['val_lr'], cfg['val_hr'], cfg['img_size'])

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=0
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"[INFO] Train samples: {len(ds_tr)}")
    print(f"[INFO] Val samples:   {len(ds_va)}")

    # ========== 模型 ==========
    G = get_generator(cfg).to(device)
    D = DiscriminatorLite().to(device)

    # ========== 优化器 ==========
    optG = torch.optim.Adam(G.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))

    # ========== 损失函数 ==========
    gan_loss = GANLoss().to(device)
    l1 = nn.L1Loss().to(device)
    perc = PerceptualLoss().to(device)

    # 双参考 feature consistency，alpha 可在 config 里配
    feat_alpha = cfg.get("feat_alpha", 0.7)
    featc = FeatureConsistency(alpha=feat_alpha).to(device)

    # 新增 edge loss
    edge_loss_fn = EdgeLoss().to(device)

    # ========== 日志文件 ==========
    log_csv = os.path.join(cfg['out_dir'], 'logs', 'train_log.csv')
    if not os.path.exists(log_csv):
        with open(log_csv, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'time',
                'loss_D', 'loss_G',
                'loss_GAN', 'loss_L1', 'loss_LPIPS', 'loss_FEAT', 'loss_EDGE',
                'w_adv', 'w_l1', 'w_lpips', 'w_feat', 'w_edge'
            ])

    met_csv = os.path.join(cfg['out_dir'], 'metrics', 'val_metrics.csv')
    if not os.path.exists(met_csv):
        with open(met_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'PSNR', 'SSIM', 'NIQE(optional)'])

    # ========== best model 记录 ==========
    best_psnr = -1e9
    best_ssim = -1e9

    # ========== 训练 ==========
    for ep in range(1, cfg['epochs'] + 1):
        G.train()
        D.train()

        # 是否启用分阶段动态权重（消融实验需要）
        use_dynamic_weighting = cfg.get("use_dynamic_weighting", True)

        if use_dynamic_weighting:
            stage_w = get_stage_weights(ep, cfg['epochs'], cfg)
            w_adv = stage_w['adv']
            w_l1 = stage_w['l1']
            w_lpips = stage_w['lpips']
            w_feat = stage_w['feat']
            w_edge = stage_w['edge']
        else:
            # 固定权重版本：用于消融实验
            w_adv = cfg['lambda_adv']
            w_l1 = cfg['lambda_l1']
            w_lpips = cfg['lambda_lpips']
            w_feat = cfg['lambda_feat']
            w_edge = cfg.get('lambda_edge', 0.0)

        # 记录 epoch 平均损失
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        epoch_loss_gan = 0.0
        epoch_loss_l1 = 0.0
        epoch_loss_lpips = 0.0
        epoch_loss_feat = 0.0
        epoch_loss_edge = 0.0
        num_batches = 0

        pbar = tqdm(dl_tr, desc=f"[Train] Epoch {ep}")

        for lr_img, hr_img in pbar:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # -------------------------
            # 1) 更新判别器 D
            # -------------------------
            fake_detach = G(lr_img).detach()

            loss_d_real = gan_loss(D(hr_img), True)
            loss_d_fake = gan_loss(D(fake_detach), False)
            loss_d = loss_d_real + loss_d_fake

            optD.zero_grad()
            loss_d.backward()
            optD.step()

            # -------------------------
            # 2) 更新生成器 G
            # -------------------------
            fake = G(lr_img)

            # 各项损失（新版）
            loss_g_gan = gan_loss(D(fake), True) * w_adv
            loss_g_l1 = l1(fake, hr_img) * w_l1
            loss_g_lpips = perc(fake, hr_img) * w_lpips

            # 期刊版：双参考 feature consistency
            loss_g_feat = featc(fake, lr_img, hr_img) * w_feat

            # 新增：edge-aware loss
            loss_g_edge = edge_loss_fn(fake, hr_img) * w_edge

            loss_g = (
                loss_g_gan
                + loss_g_l1
                + loss_g_lpips
                + loss_g_feat
                + loss_g_edge
            )

            optG.zero_grad()
            loss_g.backward()
            optG.step()

            # 统计
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            epoch_loss_gan += loss_g_gan.item()
            epoch_loss_l1 += loss_g_l1.item()
            epoch_loss_lpips += loss_g_lpips.item()
            epoch_loss_feat += loss_g_feat.item()
            epoch_loss_edge += loss_g_edge.item()
            num_batches += 1

            pbar.set_postfix({
                'D': f"{loss_d.item():.3f}",
                'G': f"{loss_g.item():.3f}",
                'feat': f"{loss_g_feat.item():.3f}",
                'edge': f"{loss_g_edge.item():.3f}"
            })

        # epoch 平均
        avg_loss_d = epoch_loss_d / max(num_batches, 1)
        avg_loss_g = epoch_loss_g / max(num_batches, 1)
        avg_loss_gan = epoch_loss_gan / max(num_batches, 1)
        avg_loss_l1 = epoch_loss_l1 / max(num_batches, 1)
        avg_loss_lpips = epoch_loss_lpips / max(num_batches, 1)
        avg_loss_feat = epoch_loss_feat / max(num_batches, 1)
        avg_loss_edge = epoch_loss_edge / max(num_batches, 1)

        # -------------------------
        # 3) 记录训练日志
        # -------------------------
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                ep, int(time.time()),
                f"{avg_loss_d:.6f}",
                f"{avg_loss_g:.6f}",
                f"{avg_loss_gan:.6f}",
                f"{avg_loss_l1:.6f}",
                f"{avg_loss_lpips:.6f}",
                f"{avg_loss_feat:.6f}",
                f"{avg_loss_edge:.6f}",
                f"{w_adv:.6f}",
                f"{w_l1:.6f}",
                f"{w_lpips:.6f}",
                f"{w_feat:.6f}",
                f"{w_edge:.6f}",
            ])

        # -------------------------
        # 4) 验证：样图 + 指标
        # -------------------------
        G.eval()
        with torch.no_grad():
            # 保存一张样图
            for i, (lr_img, hr_img) in enumerate(dl_va):
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                fake = G(lr_img)
                save_image_grid(
                    fake,
                    os.path.join(cfg['out_dir'], 'samples', f'ep{ep:03d}.png')
                )
                break

            # 验证指标：目前仍按你原来的抽样逻辑
            # 之后如果你愿意，我们再单独写完整 test_metrics.py
            psnrs, ssims, niqes = [], [], []

            for i, (lr_img, hr_img) in enumerate(dl_va):
                if i >= 5:
                    break
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                fake = G(lr_img)

                for k in range(fake.size(0)):
                    p, s = measure_psnr_ssim(fake[k], hr_img[k])
                    psnrs.append(p)
                    ssims.append(s)

                    ni = try_niqe(tensor_to_img255(fake[k]))
                    if ni is not None:
                        niqes.append(ni)

            avg_psnr = sum(psnrs) / len(psnrs) if psnrs else 0.0
            avg_ssim = sum(ssims) / len(ssims) if ssims else 0.0
            avg_niqe = sum(niqes) / len(niqes) if niqes else None

            with open(met_csv, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    ep,
                    f"{avg_psnr:.4f}" if psnrs else "",
                    f"{avg_ssim:.4f}" if ssims else "",
                    f"{avg_niqe:.4f}" if avg_niqe is not None else ""
                ])

        print(f"[VAL] Epoch {ep}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, NIQE={avg_niqe if avg_niqe is not None else 'N/A'}")

        # -------------------------
        # 5) 保存 best / interval / last
        # -------------------------
        ckpt_dir = os.path.join(cfg['out_dir'], 'checkpoints')

        # best PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'best_psnr_G.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'best_psnr_D.pth'))
            print(f"[SAVE] New best PSNR model saved: {best_psnr:.4f}")

        # best SSIM
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'best_ssim_G.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'best_ssim_D.pth'))
            print(f"[SAVE] New best SSIM model saved: {best_ssim:.4f}")

        # 定期保存
        if ep % cfg['save_interval'] == 0:
            torch.save(G.state_dict(), os.path.join(ckpt_dir, f'G_ep{ep}.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, f'D_ep{ep}.pth'))

        # last
        torch.save(G.state_dict(), os.path.join(ckpt_dir, 'last_G.pth'))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, 'last_D.pth'))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='config.yaml')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    main(args.cfg, args.device)