import os
import csv
import time
import yaml
import argparse

import torch


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


def count_parameters(model):
    """
    统计可训练参数量
    返回：
        total_params: 参数个数（整数）
        total_params_m: 参数量（百万）
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_m = total_params / 1e6
    return total_params, total_params_m


@torch.no_grad()
def measure_fps_and_latency(model, device, img_size=256, warmup=50, runs=200):
    """
    测量单张推理速度：
    - 先 warmup
    - 再正式计时
    - batch size 固定为 1，更适合论文中写单张实时推理

    返回：
        fps
        latency_ms
    """
    model.eval()

    x = torch.randn(1, 3, img_size, img_size).to(device)

    # 预热，避免第一次调用偏慢
    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(runs):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start

    avg_time = total_time / runs          # 单张平均秒数
    latency_ms = avg_time * 1000.0        # 单张平均毫秒
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    return fps, latency_ms


def save_result_csv(save_path, model_name, ckpt_path, params_m, fps, latency_ms):
    """
    保存结果到 CSV
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    file_exists = os.path.exists(save_path)
    with open(save_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "checkpoint", "Params(M)", "FPS", "Latency(ms)"])
        writer.writerow([
            model_name,
            ckpt_path if ckpt_path is not None else "random_init",
            f"{params_m:.4f}",
            f"{fps:.4f}",
            f"{latency_ms:.4f}",
        ])


def main(cfg_path, ckpt_path, device, model_name_override=None):
    # ========= 读取配置 =========
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ========= 模型 =========
    model = get_generator(cfg).to(device)

    # ========= 加载权重（可选） =========
    if ckpt_path is not None and ckpt_path.strip() != "":
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("[INFO] No checkpoint provided, using random initialized model.")

    # ========= 参数量 =========
    total_params, total_params_m = count_parameters(model)

    # ========= FPS / Latency =========
    fps, latency_ms = measure_fps_and_latency(
        model=model,
        device=device,
        img_size=cfg["img_size"],
        warmup=50,
        runs=200
    )

    # ========= 模型名 =========
    if model_name_override is not None:
        model_name = model_name_override
    else:
        model_name = cfg.get("generator_type", "mobile")

    print("\n========== Efficiency Results ==========")
    print(f"Model        : {model_name}")
    print(f"Params       : {total_params} ({total_params_m:.4f} M)")
    print(f"FPS          : {fps:.4f}")
    print(f"Latency (ms) : {latency_ms:.4f}")
    print("========================================\n")

    # ========= 保存 =========
    result_csv = os.path.join(cfg["out_dir"], "metrics", "fps_metrics.csv")
    save_result_csv(
        save_path=result_csv,
        model_name=model_name,
        ckpt_path=ckpt_path,
        params_m=total_params_m,
        fps=fps,
        latency_ms=latency_ms
    )
    print(f"[INFO] Results saved to: {result_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--ckpt", type=str, default="", help="模型权重路径，可为空")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--name", type=str, default="", help="模型显示名称，可选")
    args = parser.parse_args()

    main(
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        device=args.device,
        model_name_override=args.name if args.name != "" else None
    )