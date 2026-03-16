import os
import csv
import yaml
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data.dataset import PairDataset
from utils.metrics import measure_psnr_ssim, tensor_to_img255, try_niqe


def get_generator(cfg):
    """
    根据 config 中的 generator_type 选择生成器
    """
    gen_type = cfg.get("generator_type", "mobile").lower()

    if gen_type == "funie":
        from models.generator_funie import GeneratorFUNIE
        print("[INFO] Using generator: FUnIE-GAN")
        return GeneratorFUNIE()
    else:
        from models.generator_mobile import GeneratorMobile
        print("[INFO] Using generator: MobileGAN")
        return GeneratorMobile()


@torch.no_grad()
def evaluate_dataset(generator, dataloader, device):
    """
    在完整测试集上计算平均指标
    返回：
        avg_psnr, avg_ssim, avg_niqe
    """
    generator.eval()

    psnrs = []
    ssims = []
    niqes = []

    for lr_img, hr_img in tqdm(dataloader, desc="[Eval]"):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        fake = generator(lr_img)

        for i in range(fake.size(0)):
            # PSNR / SSIM
            psnr, ssim = measure_psnr_ssim(fake[i], hr_img[i])
            psnrs.append(psnr)
            ssims.append(ssim)

            # NIQE（如果当前环境支持）
            niqe = try_niqe(tensor_to_img255(fake[i]))
            if niqe is not None:
                niqes.append(niqe)

    avg_psnr = sum(psnrs) / len(psnrs) if len(psnrs) > 0 else 0.0
    avg_ssim = sum(ssims) / len(ssims) if len(ssims) > 0 else 0.0
    avg_niqe = sum(niqes) / len(niqes) if len(niqes) > 0 else None

    return avg_psnr, avg_ssim, avg_niqe


def build_dataloader(lr_dir, hr_dir, img_size, batch_size):
    """
    构建成对测试集 DataLoader
    """
    dataset = PairDataset(lr_dir, hr_dir, img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    return dataset, dataloader


def save_result_csv(save_path, dataset_name, ckpt_path, psnr, ssim, niqe):
    """
    保存测试结果到 CSV
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    file_exists = os.path.exists(save_path)
    with open(save_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["dataset", "checkpoint", "PSNR", "SSIM", "NIQE"])
        writer.writerow([
            dataset_name,
            ckpt_path,
            f"{psnr:.4f}",
            f"{ssim:.4f}",
            f"{niqe:.4f}" if niqe is not None else "N/A"
        ])


def main(cfg_path, ckpt_path, dataset_name, device):
    # ========= 读取配置 =========
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ========= 根据数据集名选择路径 =========
    dataset_name = dataset_name.lower()

    if dataset_name == "uieb":
        lr_dir = cfg["val_lr"]
        hr_dir = cfg["val_hr"]
    elif dataset_name == "euvp":
        lr_dir = cfg["euvp_lr"]
        hr_dir = cfg["euvp_hr"]
    else:
        raise ValueError("dataset_name 只能是 'uieb' 或 'euvp'")

    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] LR dir : {lr_dir}")
    print(f"[INFO] HR dir : {hr_dir}")

    # ========= 数据 =========
    dataset, dataloader = build_dataloader(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"]
    )
    print(f"[INFO] Number of test pairs: {len(dataset)}")

    # ========= 模型 =========
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = get_generator(cfg).to(device)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # ========= 测试 =========
    avg_psnr, avg_ssim, avg_niqe = evaluate_dataset(model, dataloader, device)

    print("\n========== Final Results ==========")
    print(f"Dataset : {dataset_name}")
    print(f"PSNR    : {avg_psnr:.4f}")
    print(f"SSIM    : {avg_ssim:.4f}")
    print(f"NIQE    : {avg_niqe:.4f}" if avg_niqe is not None else "NIQE    : N/A")
    print("===================================\n")

    # ========= 保存结果 =========
    result_csv = os.path.join(cfg["out_dir"], "metrics", "test_metrics.csv")
    save_result_csv(
        save_path=result_csv,
        dataset_name=dataset_name,
        ckpt_path=ckpt_path,
        psnr=avg_psnr,
        ssim=avg_ssim,
        niqe=avg_niqe
    )
    print(f"[INFO] Results saved to: {result_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="生成器权重路径")
    parser.add_argument("--dataset", type=str, choices=["uieb", "euvp"], required=True, help="测试数据集")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    args = parser.parse_args()

    main(
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        dataset_name=args.dataset,
        device=args.device
    )