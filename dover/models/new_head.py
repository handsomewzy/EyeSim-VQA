import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

# 可视化保存函数
def save_feat_per_frame_flattened(feat: torch.Tensor, save_dir="vis/flatten_per_frame", prefix="feat"):
    """
    对每一帧 T 中的 [B, C, H, W] 特征图展平为 [B, C*H*W]，再 reshape 为二维图像保存。

    Args:
        feat (Tensor): shape [B, C, T, H, W]
        save_dir (str): 保存路径
        prefix (str): 文件名前缀
    """
    os.makedirs(save_dir, exist_ok=True)

    B, C, T, H, W = feat.shape
    N = C * H * W

    # 为可视化 reshape 计算合适的尺寸
    grid_h = int(np.floor(np.sqrt(N)))
    grid_w = int(np.ceil(N / grid_h))
    pad_len = grid_h * grid_w - N

    for b in range(B):
        for t in range(T):
            vec = feat[b, :, t].reshape(-1).detach().cpu().numpy()  # [C*H*W]
            if pad_len > 0:
                vec = np.concatenate([vec, np.zeros(pad_len)])
            img2d = vec.reshape(grid_h, grid_w)

            # Normalize
            img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-6)

            # Save
            plt.figure(figsize=(4, 4))
            plt.imshow(img2d, cmap='viridis')
            plt.axis('off')
            plt.title(f"{prefix}_b{b}_t{t}")
            fname = f"{prefix}_b{b}_t{t:02d}.png"
            plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight', pad_inches=0.05)
            plt.close()


def save_heatmap(tensor, save_dir, name, upsample_size=(224, 224)):
    os.makedirs(save_dir, exist_ok=True)
    tensor = F.interpolate(tensor, size=(tensor.shape[2], *upsample_size), mode='trilinear', align_corners=False)
    tensor = tensor.detach().cpu().squeeze(0).squeeze(0).numpy()  # [T, H, W]

    for t in range(tensor.shape[0]):
        frame_map = tensor[t]
        frame_map = (frame_map - np.min(frame_map)) / (np.max(frame_map) - np.min(frame_map) + 1e-6)  # normalize

        plt.figure(figsize=(4, 4))
        plt.imshow(frame_map, cmap='jet')
        plt.axis('off')
        # plt.title(f"{name} - Frame {t}")
        # plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"{name}_frame_{t:02d}.png"), bbox_inches='tight', pad_inches=0.05)
        plt.close()

    

# --- Dynamic Tanh ---
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None, None] + self.bias[:, None, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

# --- Mamba-inspired lightweight temporal module ---
class TemporalMambaBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim=32):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.proj = nn.Conv3d(hidden_dim, in_channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.proj(self.act(self.conv(x)))

# --- SE block with DyT as normalization ---
class SEWithDyT(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, 1)
        self.norm = DynamicTanh(channels // reduction, channels_last=False)
        self.fc2 = nn.Conv3d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)  # [B, C, 1, 1, 1]
        s = self.fc1(s)
        s = self.norm(s)
        s = self.fc2(s)
        return x * self.sigmoid(s)

# --- Main Head ---
class VQAHead_MambaDyT(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, pre_pool=False):
        super().__init__()
        self.pre_pool = pre_pool
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_hid = nn.Conv3d(in_channels, hidden_channels, 1)

        # Score branch
        self.mamba_block = TemporalMambaBlock(hidden_channels)
        self.fc_score = nn.Conv3d(hidden_channels, 1, 1)

        # Weight branch
        self.se_dyt = SEWithDyT(hidden_channels)
        self.fc_weight = nn.Conv3d(hidden_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_maps=False):
        # x: [B, C, T, H, W]
        if self.pre_pool:
            x = self.avg_pool(x)

        x = self.gelu(self.fc_hid(x))  # [B, hidden, T, H, W]
        
        # score path
        score_feat = self.mamba_block(self.dropout(x))       # [B, hidden, T, H, W]
        score_map = self.fc_score(score_feat)                # [B, 1, T, H, W]

        # weight path
        weight_feat = self.se_dyt(self.dropout(x))           # [B, hidden, T, H, W]
        weight_map = self.sigmoid(self.fc_weight(weight_feat))  # [B, 1, T, H, W]

        # 保存weight map和score map可视化对比 假设 batch size = 1，仅在推理或验证阶段用
        # save_heatmap(score_map, save_dir="./vis/score_map", name="score")
        # save_heatmap(weight_map, save_dir="./vis/weight_map", name="weight")
        
        # save_feat_per_frame_flattened(score_feat, save_dir="./vis/score_map", prefix="score")
        # save_feat_per_frame_flattened(score_feat, save_dir="./vis/weight_map", prefix="weight")
        
        # fusion
        weighted_score_map = score_map * weight_map          # [B, 1, T, H, W]
        score = weighted_score_map.sum(dim=(-3, -2, -1)) / (weight_map.sum(dim=(-3, -2, -1)) + 1e-6)
        if return_maps:
            return score, score_map, weight_map
        else:
            return score  # [B, 1]
