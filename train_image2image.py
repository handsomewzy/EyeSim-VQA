import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import random
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as compute_psnr
# from skimage.metrics import structural_similarity as compute_ssim
import wandb
import os.path as osp
from PIL import Image
import dover.models as models
import dover.datasets as datasets
import cv2
import torch
import numpy as np
from einops import rearrange
from torchvision.transforms.functional import resize
import torchvision
import pickle


def get_network(name, pretrained=False):
    network = {
        "VGG16": torchvision.models.vgg16(pretrained=pretrained),
        "VGG16_bn": torchvision.models.vgg16_bn(pretrained=pretrained),
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
	"resnet101": torchvision.models.resnet101(pretrained=pretrained),
	"resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in network.keys():
        raise KeyError(f"{name} is not a valid network architecture")
    return network[name]


def load_contrique(model_path, regressor_path, device):
    encoder = get_network('resnet50', pretrained=False)
    model = models.CONTRIQUE_model(None, encoder, 2048)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    with open(regressor_path, 'rb') as f:
        regressor = pickle.load(f)

    return model, regressor


def compute_contrique_score(tensor_5d, contrique_model, contrique_regressor, time_indices=[0, 4, 8]):
    """
    从视频片段中提取若干帧并计算 CONTRIQUE 质量得分。
    
    Args:
        tensor_5d (Tensor): [B, C, T, H, W]
        contrique_model (nn.Module): CONTRIQUE 预训练模型（torch模块）
        contrique_regressor (sklearn-like): CONTRIQUE 回归器
        time_indices (list): 时间维上的帧索引列表
    
    Returns:
        Tensor: CONTRIQUE 分数（[B] 或 [B, 1]），可用于 loss 正则化
    """
    B, C, T, H, W = tensor_5d.shape
    device = tensor_5d.device
    scores = []

    contrique_model.eval()

    with torch.no_grad():
        for b in range(B):
            feats = []
            for t in time_indices:
                t = min(t, T - 1)
                frame = tensor_5d[b, :, t, :, :].unsqueeze(0)  # [1, C, H, W]
                frame_down = resize(frame, [H // 2, W // 2])

                _,_, _, _, feat1, feat2, _, _ = contrique_model(frame, frame_down)
                feat_cat = torch.cat([feat1, feat2], dim=1).cpu().numpy()  # [1, dim]
                feats.append(feat_cat)

            feats = np.mean(np.stack(feats), axis=0)  # [1, dim]
            score = contrique_regressor.predict(feats)[0]
            scores.append(score)

    scores = torch.tensor(scores, dtype=torch.float32, device=device)
    return scores  # shape: [B]




class DenoisingWithIdentityLoss(nn.Module):
    def __init__(self, loss_type="charbonnier", lambda_identity=0.1):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_identity = lambda_identity

    def charbonnier(self, diff):
        return torch.sqrt(diff ** 2 + 1e-6)

    def forward(self, pred_noisy, target_clean, pred_clean=None, clean_input=None):
        # 主损失（噪声图 → 原图）
        if self.loss_type == "charbonnier":
            main_loss = self.charbonnier(pred_noisy - target_clean).mean()
        elif self.loss_type == "l1":
            main_loss = torch.nn.functional.l1_loss(pred_noisy, target_clean)
        elif self.loss_type == "mse":
            main_loss = torch.nn.functional.mse_loss(pred_noisy, target_clean)
        else:
            raise ValueError("Unsupported loss type")

        # 正则项：模型在 clean 上的重建损失
        if pred_clean is not None and clean_input is not None:
            if self.loss_type == "charbonnier":
                identity_loss = self.charbonnier(pred_clean - clean_input).mean()
            elif self.loss_type == "l1":
                identity_loss = torch.nn.functional.l1_loss(pred_clean, clean_input)
            elif self.loss_type == "mse":
                identity_loss = torch.nn.functional.mse_loss(pred_clean, clean_input)
            else:
                identity_loss = 0.0
        else:
            identity_loss = 0.0

        return main_loss + self.lambda_identity * identity_loss


def strip_prefix_from_state_dict(state_dict, prefix="generator."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # 去除前缀
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v  # 保留无前缀的键
    return new_state_dict


def compute_psnr_float32(img1, img2, data_range=255.0):
    """
    精确计算 float32 格式图像的 PSNR。

    Args:
        img1 (np.ndarray): 原图 [H, W, C] float32, 范围 [0, 255]
        img2 (np.ndarray): 重建图像 [H, W, C] float32, 范围 [0, 255]

    Returns:
        float: PSNR 值（单位：dB）
    """
    assert img1.shape == img2.shape, "图像尺寸不一致"
    assert img1.dtype == np.float32 and img2.dtype == np.float32, "必须为 float32 类型"
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def compute_ssim_float32(img1, img2, data_range=255.0):
    """
    精确计算 float32 格式图像的 SSIM（结构相似性）。

    Args:
        img1 (np.ndarray): 原图 [H, W, C] float32, 范围 [0, 255]
        img2 (np.ndarray): 重建图像 [H, W, C] float32, 范围 [0, 255]

    Returns:
        float: SSIM 值（范围 0 ~ 1）
    """
    assert img1.shape == img2.shape, "图像尺寸不一致"
    assert img1.dtype == np.float32 and img2.dtype == np.float32, "必须为 float32 类型"

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    ssim_total = 0.0
    for c in range(img1.shape[2]):
        x = img1[:, :, c]
        y = img2[:, :, c]

        mu_x = cv2.filter2D(x, -1, window)
        mu_y = cv2.filter2D(y, -1, window)

        sigma_x = cv2.filter2D(x * x, -1, window) - mu_x ** 2
        sigma_y = cv2.filter2D(y * y, -1, window) - mu_y ** 2
        sigma_xy = cv2.filter2D(x * y, -1, window) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        ssim_total += ssim_map.mean()

    return ssim_total / img1.shape[2]


def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    print(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split(",")
            filename, _, _, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )

# ✅ Charbonnier Loss
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        loss = torch.sqrt((pred - target) ** 2 + self.eps)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ✅ 数据加载函数（根据 opt["data"] 字典构造）
def build_dataloaders(opt, split=0):
    if opt.get("split_seed", -1) > 0:
        opt["data"]["train"] = deepcopy(opt["data"][opt["target_set"]])
        opt["data"]["eval"] = deepcopy(opt["data"][opt["target_set"]])
        split_duo = train_test_split(
            opt["data"][opt["target_set"]]["args"]["data_prefix"],
            opt["data"][opt["target_set"]]["args"]["anno_file"],
            ratio=0.99,
            seed=opt["split_seed"] * (split + 1),
        )
        opt["data"]["train"]["args"]["anno_file"], opt["data"]["eval"]["args"]["anno_file"] = split_duo
        opt["data"]["train"]["args"]["sample_types"]["technical"]["num_clips"] = 1

    # ✅ Train dataset 构造
    train_datasets = {}
    for key in opt["data"]:
        if key.startswith("train"):
            ds_class = getattr(datasets, opt["data"][key]["type"])
            train_dataset = ds_class(opt["data"][key]["args"])
            num_samples = len(train_dataset)
            print(f"✅ Train Set [{key}]: {num_samples} samples")
            if num_samples == 0:
                print(f"⚠️ WARNING: train dataset {key} is empty, skipping...")
                continue
            train_datasets[key] = train_dataset

    # ✅ Train loader 构造
    train_loaders = {
        key: torch.utils.data.DataLoader(
            ds,
            batch_size=opt["batch_size"],
            num_workers=opt["num_workers"],
            shuffle=True,
            drop_last=True  # 避免 batch=1 导致 BN 崩溃（可选）
        )
        for key, ds in train_datasets.items()
    }

    # ✅ Eval dataset 构造
    val_datasets = {}
    for key in opt["data"]:
        if key.startswith("eval"):
            ds_class = getattr(datasets, opt["data"][key]["type"])
            val_dataset = ds_class(opt["data"][key]["args"])
            num_samples = len(val_dataset)
            print(f"✅ Eval Set [{key}]: {num_samples} samples")
            if num_samples == 0:
                print(f"⚠️ WARNING: eval dataset {key} is empty, skipping...")
                continue
            val_datasets[key] = val_dataset

    # ✅ Eval loader 构造
    val_loaders = {
        key: torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            num_workers=opt["num_workers"],
            shuffle=False,
            pin_memory=True
        )
        for key, ds in val_datasets.items()
    }

    return train_loaders, val_loaders


# ✅ 训练过程
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, branch="aesthetic"):
    model.train()
    total_loss = 0
    
    IQA_model, IQA_regressor = load_contrique(model_path="/data1/userhome/luwen/Code/wzy/DOVER-master/CONTRIQUE/CONTRIQUE_checkpoint25.tar",
                                        regressor_path="/data1/userhome/luwen/Code/wzy/DOVER-master/CONTRIQUE/models/Koniq.save",
                                        device=device)

    for i, data in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        # "technical" / "aesthetic"
        src = data["distorted_video"][branch].to(device)
        tgt = data["orig_video"][branch].to(device)
        
        # # 可视化输入图像
        # import os
        # import torchvision.utils as vutils
        # from PIL import Image

        # save_vis_dir = "/data1/userhome/luwen/Code/wzy/DOVER-master/vis_test_0"  # 修改为你希望保存的位置
        # os.makedirs(save_vis_dir, exist_ok=True)

        # # vclips[key]: Tensor of shape [B, C, T, H, W]
        # clip = src.detach().cpu()  # 确保在 CPU 上
        # B, C, T, H, W = clip.shape

        # for b in range(B):
        #     for t in range(T):
        #         frame = clip[b, :, t]  # [C, H, W]
        #         frame = (frame * 255).clamp(0, 255).byte()
        #         frame_np = frame.permute(1, 2, 0).numpy()  # [H, W, C]
        #         img = Image.fromarray(frame_np)
        #         img.save(os.path.join(save_vis_dir, f"b{b}_t{t}.png"))
        # asd
        
        pred = model(src.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        pred_tgt = model(src.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        loss_1 = loss_fn(pred, tgt)
        loss_2 = 0.3 * loss_fn(pred_tgt, tgt)
        loss = loss_1 + loss_2
        
        IQA_loss = 0.01 * (100 - compute_contrique_score(pred_tgt, IQA_model, IQA_regressor)).mean()
        loss = loss + IQA_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 🟡 每 N step 记录一次 loss 和 lr
        # if (i + 1) % 1 == 0:
        wandb.log({
            "train/loss_1": loss_1.item(),
            "train/loss_2": loss_2.item(),
            "train/IQA_loss": IQA_loss.item(),
            "train/lr": optimizer.param_groups[0]["lr"]
        })
        # if i == 5:
        #     break

    avg_loss = total_loss / len(loader)
    print(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.6f}")
    wandb.log({"train/avg_epoch_loss": avg_loss})
    
    
# ✅ PSNR / SSIM 测试函数
def evaluate_model(model, dataloader, device, save_dir=None, save_model_path=None, best_iqa=0, branch="aesthetic"):
    model.eval()
    psnr_list, ssim_list, iqa_pred_list = [], [], []

    # 加载 CONTRIQUE 模型和回归器
    IQA_model, IQA_regressor = load_contrique(
        model_path="/data1/userhome/luwen/Code/wzy/DOVER-master/CONTRIQUE/CONTRIQUE_checkpoint25.tar",
        regressor_path="/data1/userhome/luwen/Code/wzy/DOVER-master/CONTRIQUE/models/Koniq.save",
        device=device
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader, desc="Evaluating")):
            tgt = data["orig_video"][branch].to(device)  # [B, C, T, H, W]
            pred = model(tgt.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

            # CONTRIQUE IQA 分数（越高越好）
            iqa_pred = compute_contrique_score(pred, IQA_model, IQA_regressor)  # tensor[B]
            iqa_pred_list.extend(iqa_pred.detach().cpu().tolist())

            B, C, T, H, W = pred.shape
            for b in range(B):
                for t in range(T):
                    pred_frame = (pred[b, :, t] * 255).clamp(0, 255).cpu().permute(1, 2, 0).numpy().astype(np.float32)
                    tgt_frame = (tgt[b, :, t] * 255).clamp(0, 255).cpu().permute(1, 2, 0).numpy().astype(np.float32)

                    psnr = compute_psnr_float32(tgt_frame, pred_frame)
                    ssim = compute_ssim_float32(tgt_frame, pred_frame)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    if save_dir and b == 0 and t < 3:
                        Image.fromarray(pred_frame.astype(np.uint8)).save(os.path.join(save_dir, f"pred_{idx}_t{t}.png"))
                        Image.fromarray(tgt_frame.astype(np.uint8)).save(os.path.join(save_dir, f"gt_{idx}_t{t}.png"))

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_iqa = np.mean(iqa_pred_list)

    print(f"📈 PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | IQA: {avg_iqa:.4f}")
    wandb.log({
        "eval/psnr": avg_psnr,
        "eval/ssim": avg_ssim,
        "eval/iqa": avg_iqa
    })

    # ✅ 保存最佳模型（依据 IQA 分数）
    if save_model_path is not None and avg_iqa > best_iqa:
        torch.save(model.state_dict(), save_model_path)
        print(f"✅ 模型已保存（IQA 最优）至: {save_model_path}")
        best_iqa = avg_iqa

    return avg_psnr, avg_ssim, best_iqa

# ✅ 主函数
def main():
    # "technical" / "aesthetic"
    branch = "technical"
    # 🟡 在 main() 开始处加入：
    wandb.init(
        project="vqa-cleannet-image2image-{}".format(branch),  # 你的项目名
        name="run_with_distortion_restoration",  # 当前实验名
        config={
            "optimizer": "Adam",
            "lr": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "loss": "Charbonnier",
            "epochs": 10,
            "batch_size": 8
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = {
        "target_set": "val-kv1k",
        "split_seed": 42,
        "batch_size": 4,  # 如果仍然OOM，可尝试降为2
        "num_workers": 4,
        "data": {
            "val-kv1k": {
                "type": "ViewDecompositionDataset_VE",
                "args": {
                    "phase": "train",
                    "anno_file": "./examplar_data_labels/KoNViD/labels.txt",
                    "data_prefix": "/data1/userhome/luwen/Code/wzy/VQA_dataset",
                    "sample_types": {
                        "technical": {
                            "fragments_h": 5,        # 降低碎片数
                            "fragments_w": 5,
                            "fsize_h": 16,           # 降低每个碎片大小
                            "fsize_w": 16,
                            "aligned": 16,           # 对齐更小尺寸
                            "clip_len": 16,          # 时间长度减半
                            "frame_interval": 2,
                            "num_clips": 1,          # 减少clip数量
                        },
                        "aesthetic": {
                            "size_h": 112,           # 空间分辨率减半
                            "size_w": 112,
                            "clip_len": 16,          # 时间长度减半
                            "frame_interval": 2,
                            "t_frag": 16,            # 时间碎片减半
                            "num_clips": 1,
                        }
                    }
                }
            }
        }
    }

    # 加载数据
    train_loaders, val_loaders = build_dataloaders(opt)

    # 初始化模型
    # model = models.BasicVSRNet().to(device)
    
    model = models.CleanNet().to(device)
    ckpt = torch.load("/data1/userhome/luwen/Code/wzy/CAD2VSR/realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds_20211027-0e2ff207.pth", map_location="cpu")
    model.load_state_dict(strip_prefix_from_state_dict(ckpt['state_dict'], prefix="generator."), strict=False)

    # base_model = models.FastDVDnet(num_input_frames=4).to(device)
    # model = models.FastDVDnetWrapper(base_model, temp_psz=5, noise_std=25/255.).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400000, eta_min=1e-7)
    loss_fn = CharbonnierLoss()

    # 训练 + 测试
    best_iqa = 0
    for epoch in range(1, 31):
        for _, train_loader in train_loaders.items():
            train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch,branch)
        scheduler.step()

        for key, val_loader in val_loaders.items():
            print(f"\n🔍 Evaluation on {key}")
            psnr, ssim, best_psnr = evaluate_model(
                model,
                val_loader,
                device,
                save_dir="./eval_outputs_{}/".format(branch),
                save_model_path="./checkpoints/best_model_{}.pth".format(branch),
                best_iqa=best_iqa,
                branch=branch
            )


if __name__ == "__main__":
    main()