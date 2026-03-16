import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import torchvision.models as models


class GANLoss(nn.Module):
    """
    对抗损失：
    - 使用 BCEWithLogitsLoss
    - 判别器输出不需要额外过 sigmoid
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, is_real: bool):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.bce(pred, target)


class PerceptualLoss(nn.Module):
    """
    感知损失（LPIPS）：
    - 输入期望范围为 [-1, 1]
    - LPIPS 返回形状通常为 [N,1,1,1]
    - 这里统一取 mean 得到标量
    """
    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net='vgg')

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.lpips(x, y).mean()


class FeatureConsistency(nn.Module):
    """
    识别感知特征一致性损失（期刊改进版）：
    使用冻结的 ResNet18 提取高层语义特征，约束增强图像在语义空间中：

    1. 接近高质量参考图像（target alignment）
    2. 不偏离原始退化输入图像（content preservation）

    最终形式：
        L_feat = alpha * ||phi(fake) - phi(hr)||^2
               + (1-alpha) * ||phi(fake) - phi(lr)||^2

    兼容两种调用方式：
    - 旧版：forward(fake, lr)
      仅计算 fake vs lr
    - 新版：forward(fake, lr, hr)
      计算双参考一致性损失

    参数：
    - alpha: 高质量参考项权重，默认 0.7
             也就是更偏向“向参考图靠拢”，但仍保留输入内容约束
    """
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha

        # 加载 ImageNet 预训练的 ResNet18，并去掉最后 fc 层
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(model.children())[:-1]).eval()  # [N, 512, 1, 1]

        # 冻结参数
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ImageNet 标准化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入从 [-1,1] 映射到 [0,1]，再做 ImageNet 标准化
        """
        x01 = (x.clamp(-1, 1) * 0.5) + 0.5
        return (x01 - self.mean) / self.std

    def _extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取 ResNet18 全局语义特征
        输出形状: [N, 512]
        """
        x = self._preprocess(x)
        feat = self.backbone(x).flatten(1)
        return feat

    def forward(
        self,
        fake: torch.Tensor,
        lr: torch.Tensor,
        hr: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        参数：
        - fake: 生成器输出增强图
        - lr:   原始退化图
        - hr:   高质量参考图（可选）

        返回：
        - 标量损失

        调用逻辑：
        1) 如果 hr is None:
           使用旧版单参考形式：MSE(phi(fake), phi(lr))

        2) 如果 hr is not None:
           使用新版双参考形式：
           alpha * MSE(phi(fake), phi(hr))
           + (1-alpha) * MSE(phi(fake), phi(lr))
        """
        f_fake = self._extract_feature(fake)
        f_lr = self._extract_feature(lr)

        # 旧版兼容：只对齐输入图
        if hr is None:
            return torch.mean((f_fake - f_lr) ** 2)

        # 新版：双参考约束
        f_hr = self._extract_feature(hr)

        loss_target = torch.mean((f_fake - f_hr) ** 2)  # 向高质量目标域靠拢
        loss_source = torch.mean((f_fake - f_lr) ** 2)  # 保留原始场景语义

        return self.alpha * loss_target + (1.0 - self.alpha) * loss_source


class EdgeLoss(nn.Module):
    """
    边缘一致性损失（新增）：
    - 使用 Sobel 算子提取边缘
    - 约束增强图与参考图的梯度/轮廓一致性
    - 只影响训练，不增加推理参数量

    公式近似：
        L_edge = ||grad(fake) - grad(hr)||_1
    """
    def __init__(self):
        super().__init__()

        # Sobel X
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Sobel Y
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _rgb_to_gray(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 RGB 转为灰度：
        输入: [N,3,H,W]
        输出: [N,1,H,W]
        """
        if x.shape[1] == 1:
            return x
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray

    def _gradient_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算梯度幅值图
        """
        x_gray = self._rgb_to_gray(x)

        gx = F.conv2d(x_gray, self.sobel_x, padding=1)
        gy = F.conv2d(x_gray, self.sobel_y, padding=1)

        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return grad

    def forward(self, fake: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        grad_fake = self._gradient_map(fake)
        grad_hr = self._gradient_map(hr)
        return F.l1_loss(grad_fake, grad_hr)


if __name__ == "__main__":
    # 简单自检：保证各模块能正常前向
    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
    z = torch.randn(2, 3, 256, 256)

    gan_loss = GANLoss()
    perceptual_loss = PerceptualLoss()
    feat_loss = FeatureConsistency(alpha=0.7)
    edge_loss = EdgeLoss()

    # GAN loss
    pred = torch.randn(2, 1, 16, 16)
    print("GAN real:", gan_loss(pred, True).item())
    print("GAN fake:", gan_loss(pred, False).item())

    # LPIPS
    print("LPIPS:", perceptual_loss(x, y).item())

    # Feature consistency
    print("Feat old:", feat_loss(x, y).item())       # 旧版兼容
    print("Feat new:", feat_loss(x, y, z).item())    # 新版双参考

    # Edge loss
    print("Edge:", edge_loss(x, y).item())