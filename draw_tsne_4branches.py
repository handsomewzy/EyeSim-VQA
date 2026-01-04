import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def bin_mos_scores(mos_array):
    """
    将 MOS 分数划分为 5 个区间 [0,1), [1,2), [2,3), [3,4), [4,5]
    返回整数标签
    """
    return np.clip(mos_array.astype(int), 0, 4)

import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.manifold import TSNE
from matplotlib import cm
def plot_tsne(features, mos, title, save_path):
    """
    features: [N, D] PCA-reduced features
    mos: [N] continuous scores
    """
    # 自动设置 perplexity
    perplexity = min(32, max(2, (features.shape[0] - 1) // 3))
    print(f"[t-SNE] Reducing {features.shape[0]} samples from dim {features.shape[1]} to 2D... (perplexity={perplexity})")

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(features)

    def bin_mos_scores(mos_array):
        bins = [0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.1]  # 右边界开一点防止 float==5 出错
        return np.digitize(mos_array, bins) - 1  # 使 bin 结果从 0 开始


    mos_bins = bin_mos_scores(mos)  # [N], 0~4
    labels = ['MOS ∈ 0–2', 'MOS ∈ 2–2.5', 'MOS ∈ 2.5–3', 'MOS ∈ 3–3.5', 'MOS ∈ 3.5–4', 'MOS ∈ 4–5']
    colors = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4', '#542788']  # 多一组颜色
    
    # 绘图
    plt.figure(figsize=(6, 5))
    for i in range(6):
        idx = (mos_bins == i)
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1],
                    label=labels[i], color=colors[i], s=25, alpha=0.6, edgecolors='none')

    # 图例样式
    plt.legend(
        loc='lower right',
        fontsize=10,
        frameon=True,
        facecolor='white',   # 白色背景
        edgecolor='black',   # 黑边框（可选）
        framealpha=1.0       # 完全不透明
    )
    
    plt.title(title, fontsize=14, pad=10)
    plt.xticks([]); plt.yticks([])  # 去掉坐标轴刻度
    plt.tight_layout()

    # 保存图像（双格式）
    plt.savefig(save_path, dpi=500)
    if save_path.endswith('.png'):
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=500)
    plt.close()
    print(f"✅ Saved: {save_path}")

def main():
    input_dir = "tsne_features_4branches"
    os.makedirs("tsne_plots", exist_ok=True)

    branches = [
        ("feat_aes_pca.npy", "mos_feat_aes.npy", "t-SNE results of enhanced aesthetic branch features", "tsne_feat_aes.png"),
        ("clean_aes_pca.npy", "mos_clean_aes.npy", "t-SNE results of raw aesthetic branch features", "tsne_clean_aes.png"),
        ("feat_tec_pca.npy", "mos_feat_tec.npy", "t-SNE results of raw technical branch features", "tsne_feat_tec.png"),
        ("vsr_tec_pca.npy", "mos_vsr_tec.npy", "t-SNE results of enhanced technical branch features", "tsne_vsr_tec.png"),
    ]

    for feat_file, mos_file, title, save_name in branches:
        feat_path = os.path.join(input_dir, feat_file)
        mos_path = os.path.join(input_dir, mos_file)
        save_path = os.path.join("tsne_plots", save_name)

        features = np.load(feat_path)
        mos = np.load(mos_path)
        # 映射到 [0, 5]
        mos_min, mos_max = mos.min(), mos.max()
        mos_mapped = 5 * (mos - mos_min) / (mos_max - mos_min)

        plot_tsne(features, mos_mapped, title, save_path)

if __name__ == "__main__":
    main()
