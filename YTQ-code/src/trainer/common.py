import torch
import torch.nn.functional as F
from torch import distributed as dist


class timer:
    def __init__(self):
        self.acc = 0
        self.t0 = torch.cuda.Event(enable_timing=True)
        self.t1 = torch.cuda.Event(enable_timing=True)
        self.tic()

    def tic(self):
        self.t0.record()

    def toc(self, restart=False):
        self.t1.record()
        torch.cuda.synchronize()
        diff = self.t0.elapsed_time(self.t1) / 1000.0
        if restart:
            self.tic()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def reduce_loss_dict(loss_dict, world_size):
    if world_size == 1:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = dict(zip(keys, losses))
    return reduced_losses

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, 3, H, W] in [-1, 1] (AOT-GAN默认就是这个范围)
    return: [B, 1, H, W]
    """
    if x.size(1) == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


# def sobel_edge(x: torch.Tensor, thr: float = 0.1) -> torch.Tensor:
#     """
#     用 Sobel 从 GT 图像生成 edge_gt（训练监督用）。
#     - 输出是二值边缘: {0,1}
#     - 不依赖opencv/kornia，纯torch，可在GPU上跑
#     """
#     gray = rgb_to_gray(x)  # [B,1,H,W]
#
#     # Sobel kernels
#     kx = torch.tensor([[-1, 0, 1],
#                        [-2, 0, 2],
#                        [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
#     ky = torch.tensor([[-1, -2, -1],
#                        [ 0,  0,  0],
#                        [ 1,  2,  1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
#
#     gx = F.conv2d(gray, kx, padding=1)
#     gy = F.conv2d(gray, ky, padding=1)
#
#     mag = torch.sqrt(gx * gx + gy * gy + 1e-12)  # [B,1,H,W]
#
#     # 归一化到[0,1]，再阈值二值化（thr 可调）
#     mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-12)
#     edge = (mag > thr).float()
#     return edge

def sobel_edge(x: torch.Tensor, thr: float = 0.1) -> torch.Tensor:
    """
    ✅ 方案A（soft target）：
    用 Sobel 从 GT 图像生成 edge_gt（训练监督用）。
    - 输出是连续的边缘强度图: [0,1]，而不是{0,1}二值
    - 不依赖opencv/kornia，纯torch，可在GPU上跑

    参数 thr 保留但在 soft 模式下不使用（避免改动其他代码调用方式）。
    """
    gray = rgb_to_gray(x)  # [B,1,H,W]

    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)

    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)  # [B,1,H,W]

    # ✅ 关键：归一化到[0,1]，作为 soft target
    # 每张图按自身最大值归一化，避免尺度漂移
    mag_norm = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-12)
    mag_norm = mag_norm.clamp(0.0, 1.0)

    return mag_norm