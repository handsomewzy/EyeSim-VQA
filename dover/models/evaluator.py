import time
from functools import partial, reduce
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool3d
import torch.nn.functional as F

from .conv_backbone import convnext_3d_small, convnext_3d_tiny, convnextv2_3d_pico, convnextv2_3d_femto
from .head import IQAHead, VARHead, VQAHead, VQAHead_DualWeight
from .new_head import VQAHead_MambaDyT
from .swin_backbone import SwinTransformer2D as ImageBackbone
from .swin_backbone import SwinTransformer3D as VideoBackbone
from .swin_backbone import swin_3d_small, swin_3d_tiny, swin_3d_mamba
from .cleannet_backbone import CleanNet, BasicVSRNet
from .gae_backbone import GAE
import random
from .dynamic_tanh import convert_ln_to_dyt


class BaseEvaluator(nn.Module):
    def __init__(
        self, backbone=dict(), vqa_head=dict(),
    ):
        super().__init__()
        self.backbone = VideoBackbone(**backbone)
        self.vqa_head = VQAHead(**vqa_head)

    def forward(self, vclip, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(vclip)
                score = self.vqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            return score

    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns


class IntraRef(nn.Module):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys="fragments,resize",
        multi=False,
        layer=-1,
        backbone=dict(
            resize={"window_size": (4, 4, 4)}, fragments={"window_size": (4, 4, 4)}
        ),
        divide_head=False,
        vqa_head=dict(in_channels=768),
        var=False,
    ):
        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        super().__init__()
        for key, hypers in backbone.items():
            print(backbone_size)
            if key not in self.backbone_preserve_keys:
                continue
            if backbone_size == "divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == "swin_tiny":
                b = swin_3d_tiny(**backbone[key])
            elif t_backbone_size == "swin_tiny_grpb":
                # to reproduce fast-vqa
                b = VideoBackbone()
                # Dyt把归一化层都去掉看看效果。
                # b = convert_ln_to_dyt(b)
            elif t_backbone_size == "swin_3d_mamba":
                b = swin_3d_mamba()
                # Dyt把归一化层都去掉看看效果。
                # b = convert_ln_to_dyt(b)
            elif t_backbone_size == "swin_tiny_grpb_m":
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4, 4, 4), frag_biases=[0, 0, 0, 0])
            elif t_backbone_size == "swin_small":
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == "conv_tiny":
                b = convnext_3d_tiny(pretrained=True)
                # Dyt把归一化层都去掉看看效果。
                # b = convert_ln_to_dyt(b)
            elif t_backbone_size == "conv_small":
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == "conv_femto":
                b = convnextv2_3d_femto(pretrained=True)
            elif t_backbone_size == "conv_pico":
                b = convnextv2_3d_pico(pretrained=True)
            elif t_backbone_size == "xclip":
                raise NotImplementedError
                # b = build_x_clip_model(**backbone[key])
            else:
                raise NotImplementedError
            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", b)
            
            # 注册模块：分别为不同 key 加入 CleanNet 或 BasicVSR
            if key == "technical":
                cleannet = CleanNet()
                ckpt = torch.load("/data1/userhome/luwen/Code/wzy/DOVER-master/checkpoints/best_model_technical.pth", map_location="cpu")
                cleannet.load_state_dict(ckpt, strict=False)
                setattr(self, key + "_cleannet", cleannet)
                del cleannet
                print("Setting cleannet:", key + "_cleannet")
            elif key == "aesthetic":
                BasicVSR = BasicVSRNet()
                ckpt = torch.load("/data1/userhome/luwen/Code/wzy/DOVER-master/checkpoints/best_model_aesthetic.pth", map_location="cpu")
                BasicVSR.load_state_dict(ckpt)
                setattr(self, key + "_vsr", BasicVSR)
                del BasicVSR
                print("Setting gae:", key + "_vsr")
            
        if divide_head:
            for key in backbone:
                pre_pool = False #if key == "technical" else True
                if key not in self.backbone_preserve_keys:
                    continue
                # b = VQAHead(pre_pool=pre_pool, **vqa_head)
                # 换一个最终的输出头，双分支，模仿MANIQA中的操作
                # b = VQAHead_DualWeight(pre_pool=pre_pool, **vqa_head)
                
                b = VQAHead_MambaDyT(pre_pool=pre_pool, **vqa_head)
                print("Setting head:", key + "_head")
                setattr(self, key + "_head", b)
        else:
            if var:
                self.vqa_head = VARHead(**vqa_head)
                print(b)
            else:
                self.vqa_head = VQAHead(**vqa_head)
                
    def tSNE_feature(
        self,
        vclips,
        inference=True,
        return_pooled_feats=False,
        return_raw_feats=False,
        reduce_scores=False,
        pooled=False,
        **kwargs
    ):
        assert (return_pooled_feats & return_raw_feats) == False, "Please only choose one kind of features to return"
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                feat_aes, clean_feat_aes = None, None
                feat_tec, vsr_feat_tec = None, None

                for key in vclips:
                    prefix = key.split("_")[0]  # e.g., "aesthetic" or "technical"

                    # 清晰增强模块
                    if hasattr(self, key + "_cleannet"):
                        module = getattr(self, key + "_cleannet")
                        module.eval()
                        with torch.no_grad():
                            clean_clip = module(vclips[key].permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                            clean_feat = getattr(self, prefix + "_backbone")(clean_clip, multi=self.multi, layer=self.layer, **kwargs)
                        if prefix == "technical":
                            clean_feat_aes = clean_feat

                    # 压缩/超分模块
                    elif hasattr(self, key + "_vsr"):
                        module = getattr(self, key + "_vsr")
                        module.eval()
                        with torch.no_grad():
                            vsr_clip = module(vclips[key].permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                            vsr_feat = getattr(self, prefix + "_backbone")(vsr_clip, multi=self.multi, layer=self.layer, **kwargs)
                        if prefix == "aesthetic":
                            vsr_feat_tec = vsr_feat

                    # 原始特征
                    feat = getattr(self, prefix + "_backbone")(vclips[key], multi=self.multi, layer=self.layer, **kwargs)
                    if prefix == "aesthetic":
                        feat_aes = feat
                    elif prefix == "technical":
                        feat_tec = feat

                return feat_aes, clean_feat_aes, feat_tec, vsr_feat_tec
      
               
               

    def forward(
        self,
        vclips,
        inference=True,
        return_pooled_feats=False,
        return_raw_feats=False,
        reduce_scores=False,
        pooled=False,
        **kwargs
    ):
        assert (return_pooled_feats & return_raw_feats) == False, "Please only choose one kind of features to return"
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                for key in vclips:
                    
                    # 可视化输入图像
                    # import os
                    # import torchvision.utils as vutils
                    # from PIL import Image

                    # save_vis_dir = "/data1/userhome/luwen/Code/wzy/DOVER-master/vis/gt"  # 修改为你希望保存的位置
                    # os.makedirs(save_vis_dir, exist_ok=True)

                    # # vclips[key]: Tensor of shape [B, C, T, H, W]
                    # clip = vclips[key].detach().cpu()  # 确保在 CPU 上
                    # B, C, T, H, W = clip.shape

                    # for b in range(B):
                    #     for t in range(T):
                    #         frame = clip[b, :, t]  # [C, H, W]
                    #         frame = (frame * 255).clamp(0, 255).byte()
                    #         frame_np = frame.permute(1, 2, 0).numpy()  # [H, W, C]
                    #         img = Image.fromarray(frame_np)
                    #         img.save(os.path.join(save_vis_dir, f"{key}_b{b}_t{t}.png"))
                    # asd
                    
                    # 调用清晰增强或压缩模块
                    if hasattr(self, key + "_cleannet"):
                        module = getattr(self, key + "_cleannet")
                        module.eval()
                        with torch.no_grad():
                            clip = module(vclips[key].permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                            # 随机采样 10 帧
                            T = clip.shape[2]
                            idx = sorted(random.sample(range(T), k=10))
                            sampled_clip = clip[:, :, idx, :, :]  # [B, C, 10, H, W]
                            concat_clip = torch.cat([vclips[key], sampled_clip], dim=2)

                    elif hasattr(self, key + "_vsr"):
                        module = getattr(self, key + "_vsr")
                        module.eval()
                        with torch.no_grad():
                            clip = module(vclips[key].permute(0, 2, 1, 3, 4))  # [B, T, C, H, W]
                            clip = clip.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
                            # 随机采样 10 帧
                            T = clip.shape[2]
                            idx = sorted(random.sample(range(T), k=10))
                            sampled_clip = clip[:, :, idx, :, :]  # [B, C, 10, H, W]
                            concat_clip = torch.cat([vclips[key], sampled_clip], dim=2)
                    
                    feat = getattr(self, key.split("_")[0] + "_backbone")(
                        concat_clip, multi=self.multi, layer=self.layer, **kwargs
                    )
                    
                    # feat = getattr(self, key.split("_")[0] + "_backbone")(
                    #     vclips[key], multi=self.multi, layer=self.layer, **kwargs
                    # )
                    if hasattr(self, key.split("_")[0] + "_head"):
                        scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                    if return_pooled_feats:
                        feats[key] = feat
                    if return_raw_feats:
                        feats[key] = feat
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1, 2, 3, 4))
            self.train()
            if return_pooled_feats or return_raw_feats:
                return scores, feats
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                
                # 可视化输入图像
                # import os
                # import torchvision.utils as vutils
                # from PIL import Image

                # save_vis_dir = f"/data1/userhome/luwen/Code/wzy/DOVER-master/vis/gt_{key}"  # 修改为你希望保存的位置
                # os.makedirs(save_vis_dir, exist_ok=True)

                # # vclips[key]: Tensor of shape [B, C, T, H, W]
                # clip = vclips[key].detach().cpu()  # 确保在 CPU 上
                # B, C, T, H, W = clip.shape

                # for b in range(B):
                #     for t in range(T):
                #         frame = clip[b, :, t]  # [C, H, W]
                #         frame = (frame * 255).clamp(0, 255).byte()
                #         frame_np = frame.permute(1, 2, 0).numpy()  # [H, W, C]
                #         img = Image.fromarray(frame_np)
                #         img.save(os.path.join(save_vis_dir, f"{key}_b{b}_t{t}.png"))
                # asd
                
                
                # 调用清晰增强或压缩模块
                if hasattr(self, key + "_cleannet"):
                    module = getattr(self, key + "_cleannet")
                    module.eval()
                    with torch.no_grad():
                        clip = module(vclips[key].permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                        # 随机采样 10 帧
                        T = clip.shape[2]
                        idx = sorted(random.sample(range(T), k=10))
                        sampled_clip = clip[:, :, idx, :, :]  # [B, C, 10, H, W]
                        concat_clip = torch.cat([vclips[key], sampled_clip], dim=2)


                elif hasattr(self, key + "_vsr"):
                    module = getattr(self, key + "_vsr")
                    module.eval()
                    with torch.no_grad():
                        clip = module(vclips[key].permute(0, 2, 1, 3, 4))  # [B, T, C, H, W]
                        clip = clip.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
                        # 随机采样 10 帧
                        T = clip.shape[2]
                        idx = sorted(random.sample(range(T), k=10))
                        sampled_clip = clip[:, :, idx, :, :]  # [B, C, 10, H, W]
                        concat_clip = torch.cat([vclips[key], sampled_clip], dim=2)

                
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    concat_clip, multi=self.multi, layer=self.layer, **kwargs
                )
                
                # feat = getattr(self, key.split("_")[0] + "_backbone")(
                #     vclips[key], multi=self.multi, layer=self.layer, **kwargs
                # )
                
                # branch_name = key.split("_")[0]  # e.g., "aesthetic" or "technical"
                # head_module = getattr(self, branch_name + "_head")

                # # 控制是否保存
                # score, score_map, weight_map = head_module(feat, return_maps=True)

                # # 保存 score_map 和 weight_map 图像
                # save_heatmap(score_map, save_dir=f"./vis/{branch_name}_score_map", name="score")
                # save_heatmap(weight_map, save_dir=f"./vis/{branch_name}_weight_map", name="weight")

                
                if hasattr(self, key.split("_")[0] + "_head"):
                    scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                else:
                    scores += [getattr(self, "vqa_head")(feat)]
                if return_pooled_feats:
                    feats[key] = feat.mean((-3, -2, -1))
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    print(scores.shape)

            if return_pooled_feats:
                return scores, feats
            return scores
        
    def forward_head(
        self,
        feats,
        inference=True,
        reduce_scores=False,
        pooled=False,
        **kwargs
    ):
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                for key in feats:
                    feat = feats[key]
                    if hasattr(self, key.split("_")[0] + "_head"):
                        scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1, 2, 3, 4))
            self.train()
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    vclips[key], multi=self.multi, layer=self.layer, **kwargs
                )
                if hasattr(self, key.split("_")[0] + "_head"):
                    scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                else:
                    scores += [getattr(self, "vqa_head")(feat)]
                if return_pooled_feats:
                    feats[key] = feat
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    print(scores.shape)

            if return_pooled_feats:
                return scores, feats
            return scores
        
        
class DOVER(nn.Module):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys="fragments,resize",
        multi=False,
        layer=-1,
        backbone=dict(
            resize={"window_size": (4, 4, 4)}, fragments={"window_size": (4, 4, 4)}
        ),
        divide_head=False,
        vqa_head=dict(in_channels=768),
        var=False,
    ):
        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        super().__init__()
        for key, hypers in backbone.items():
            print(backbone_size)
            if key not in self.backbone_preserve_keys:
                continue
            if backbone_size == "divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == "swin_tiny":
                b = swin_3d_tiny(**backbone[key])
            elif t_backbone_size == "swin_tiny_grpb":
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == "swin_tiny_grpb_m":
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4, 4, 4), frag_biases=[0, 0, 0, 0])
            elif t_backbone_size == "swin_small":
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == "conv_tiny":
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == "conv_small":
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == "conv_femto":
                b = convnextv2_3d_femto(pretrained=True)
            elif t_backbone_size == "conv_pico":
                b = convnextv2_3d_pico(pretrained=True)
            elif t_backbone_size == "xclip":
                raise NotImplementedError
                # b = build_x_clip_model(**backbone[key])
            else:
                raise NotImplementedError
            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", b)
        if divide_head:
            for key in backbone:
                pre_pool = False #if key == "technical" else True
                if key not in self.backbone_preserve_keys:
                    continue
                b = VQAHead(pre_pool=pre_pool, **vqa_head)
                print("Setting head:", key + "_head")
                setattr(self, key + "_head", b)
        else:
            if var:
                self.vqa_head = VARHead(**vqa_head)
                print(b)
            else:
                self.vqa_head = VQAHead(**vqa_head)
      

    def forward(
        self,
        vclips,
        inference=True,
        return_pooled_feats=False,
        return_raw_feats=False,
        reduce_scores=False,
        pooled=False,
        **kwargs
    ):
        assert (return_pooled_feats & return_raw_feats) == False, "Please only choose one kind of features to return"
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                for key in vclips:
                    feat = getattr(self, key.split("_")[0] + "_backbone")(
                        vclips[key], multi=self.multi, layer=self.layer, **kwargs
                    )
                    if hasattr(self, key.split("_")[0] + "_head"):
                        scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                    if return_pooled_feats:
                        feats[key] = feat
                    if return_raw_feats:
                        feats[key] = feat
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1, 2, 3, 4))
            self.train()
            if return_pooled_feats or return_raw_feats:
                return scores, feats
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    vclips[key], multi=self.multi, layer=self.layer, **kwargs
                )
                if hasattr(self, key.split("_")[0] + "_head"):
                    scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                else:
                    scores += [getattr(self, "vqa_head")(feat)]
                if return_pooled_feats:
                    feats[key] = feat.mean((-3, -2, -1))
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    print(scores.shape)

            if return_pooled_feats:
                return scores, feats
            return scores
        
    def forward_head(
        self,
        feats,
        inference=True,
        reduce_scores=False,
        pooled=False,
        **kwargs
    ):
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                for key in feats:
                    feat = feats[key]
                    if hasattr(self, key.split("_")[0] + "_head"):
                        scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1, 2, 3, 4))
            self.train()
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    vclips[key], multi=self.multi, layer=self.layer, **kwargs
                )
                if hasattr(self, key.split("_")[0] + "_head"):
                    scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                else:
                    scores += [getattr(self, "vqa_head")(feat)]
                if return_pooled_feats:
                    feats[key] = feat
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    print(scores.shape)

            if return_pooled_feats:
                return scores, feats
            return scores
                

class MinimumDOVER(nn.Module):
    def __init__(self):
        super().__init__()
        self.technical_backbone = VideoBackbone()
        self.aesthetic_backbone = convnext_3d_tiny(pretrained=True)
        self.technical_head = VQAHead(pre_pool=False, in_channels=768)
        self.aesthetic_head = VQAHead(pre_pool=False, in_channels=768)


    def forward(self,aesthetic_view, technical_view):
        self.eval()
        with torch.no_grad():
            aesthetic_score = self.aesthetic_head(self.aesthetic_backbone(aesthetic_view))
            technical_score = self.technical_head(self.technical_backbone(technical_view))
            
        aesthetic_score_pooled = torch.mean(aesthetic_score, (1,2,3,4))
        technical_score_pooled = torch.mean(technical_score, (1,2,3,4))
        return [aesthetic_score_pooled, technical_score_pooled]



class BaseImageEvaluator(nn.Module):
    def __init__(
        self, backbone=dict(), iqa_head=dict(),
    ):
        super().__init__()
        self.backbone = ImageBackbone(**backbone)
        self.iqa_head = IQAHead(**iqa_head)

    def forward(self, image, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(image)
                score = self.iqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(image)
            score = self.iqa_head(feat)
            return score

    def forward_with_attention(self, image):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(image, require_attn=True)
            score = self.iqa_head(feat)
            return score, avg_attns


def strip_prefix_from_state_dict(state_dict, prefix="generator."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # 去除前缀
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v  # 保留无前缀的键
    return new_state_dict



import matplotlib.pyplot as plt
import numpy as np

# 可视化保存函数
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