# Copyright (c) OpenMMLab. All rights reserved.
from logging import WARNING

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
# from mmagic.models.archs import PixelShufflePack, ResidualBlockNoBN
# from mmagic.models.utils import flow_warp, make_layer
# from mmagic.registry import MODELS

# from .arch_util import flow_warp, make_layer

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        normal_init, update_init_info,
                                        xavier_init)
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
# from .sr_backbone import default_init_weights


class CleanNet(BaseModule):
    def __init__(self,
                 mid_channels=64,
                 num_cleaning_blocks=20,
                 dynamic_refine_thres=255,
                 is_fix_cleaning=False,
                 is_sequential_cleaning=False):

        super().__init__()

        self.dynamic_refine_thres = dynamic_refine_thres / 255.
        self.is_sequential_cleaning = is_sequential_cleaning

        # image cleaning module
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )

        if is_fix_cleaning:  # keep the weights of the cleaning module fixed
            self.image_cleaning.requires_grad_(False)


    def forward(self, lqs, return_lqs=False):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            return_lqs (bool): Whether to return LQ sequence. Default: False.

        Returns:
            Tensor: Output HR sequence.
        """
        n, t, c, h, w = lqs.size()

        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            if self.is_sequential_cleaning:
                residues = []
                for i in range(0, t):
                    residue_i = self.image_cleaning(lqs[:, i, :, :, :])
                    lqs[:, i, :, :, :] += residue_i
                    residues.append(residue_i)
                residues = torch.stack(residues, dim=1)
            else:  # time -> batch, then apply cleaning at once
                # lqs = lqs.view(-1, c, h, w)
                lqs = lqs.reshape(-1, c, h, w)
                residues = self.image_cleaning(lqs)
                lqs = (lqs + residues).view(n, t, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break


        return lqs




def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
                 upsample_kernel: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for PixelShufflePack."""
        default_init_weights(self, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style modules.
        For modules with residual paths, using smaller std is better for
        stability and performance. We empirically use 0.1. See more details in
        "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

import torch.nn as nn

def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor
    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class RealBasicVSRNet(BaseModule):
    """RealBasicVSR network structure for real-world video super-resolution.

    Support only x4 upsampling.

    Paper:
        Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_propagation_blocks (int, optional): Number of residual blocks in
            each propagation branch. Default: 20.
        num_cleaning_blocks (int, optional): Number of residual blocks in the
            image cleaning module. Default: 20.
        dynamic_refine_thres (int, optional): Stop cleaning the images when
            the residue is smaller than this value. Default: 255.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        is_fix_cleaning (bool, optional): Whether to fix the weights of
            the image cleaning module during training. Default: False.
        is_sequential_cleaning (bool, optional): Whether to clean the images
            sequentially. This is used to save GPU memory, but the speed is
            slightly slower. Default: False.
    """

    def __init__(self,
                 mid_channels=64,
                 num_propagation_blocks=20,
                 num_cleaning_blocks=20,
                 dynamic_refine_thres=255,
                 spynet_pretrained="/data1/userhome/luwen/Code/wzy/CAD2VSR/spynet_20210409-c6c1bd09.pth",
                 is_fix_cleaning=False,
                 is_sequential_cleaning=False):

        super().__init__()

        self.dynamic_refine_thres = dynamic_refine_thres / 255.
        self.is_sequential_cleaning = is_sequential_cleaning

        # image cleaning module
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )

        if is_fix_cleaning:  # keep the weights of the cleaning module fixed
            self.image_cleaning.requires_grad_(False)

        # BasicVSR
        self.basicvsr = BasicVSRNet(mid_channels, num_propagation_blocks,
                                    spynet_pretrained)
        self.basicvsr.spynet.requires_grad_(False)

    def forward(self, lqs, return_lqs=False):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            return_lqs (bool): Whether to return LQ sequence. Default: False.

        Returns:
            Tensor: Output HR sequence.
        """
        n, t, c, h, w = lqs.size()

        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            if self.is_sequential_cleaning:
                residues = []
                for i in range(0, t):
                    residue_i = self.image_cleaning(lqs[:, i, :, :, :])
                    lqs[:, i, :, :, :] += residue_i
                    residues.append(residue_i)
                residues = torch.stack(residues, dim=1)
            else:  # time -> batch, then apply cleaning at once
                # lqs = lqs.view(-1, c, h, w)
                lqs = lqs.reshape(-1, c, h, w)
                residues = self.image_cleaning(lqs)
                lqs = (lqs + residues).view(n, t, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break

        # Super-resolution (BasicVSR)
        outputs = self.basicvsr(lqs)

        if return_lqs:
            return outputs, lqs
        else:
            return outputs


class BasicVSRNet(BaseModule):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.

    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, mid_channels=32, num_blocks=5, spynet_pretrained="/data1/userhome/luwen/Code/wzy/CAD2VSR/spynet_20210409-c6c1bd09.pth"):

        super().__init__()

        self.mid_channels = mid_channels

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        # self.upsample1 = PixelShufflePack(
        #     mid_channels, mid_channels, 2, upsample_kernel=3)
        # self.upsample2 = PixelShufflePack(
        #     mid_channels, 64, 2, upsample_kernel=3)
        
        self.upsample1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.upsample2 = nn.Conv2d(mid_channels, 64, kernel_size=3, padding=1)
        
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=1, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self._raised_warning = False

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lrs.size()
        if (h < 64 or w < 64) and not self._raised_warning:
            print_log(
                f'{self.__class__.__name__} is designed for input '
                'larger than 64x64, but the resolution of current image '
                f'is {h}x{w}. We recommend you to check your input.',
                'current', WARNING)
            self._raised_warning = True

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)

        # backward-time propagation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)

            # base = self.img_upsample(lr_curr)
            base = lr_curr  # 不再上采样，保持原尺寸
            out += base
            outputs[i] = out

        return torch.stack(outputs, dim=1)



class ResidualBlocksWithInputConv(BaseModule):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class SPyNet(BaseModule):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(BaseModule):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)




# from .dcn import ModulatedDeformConvPack
# class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
#     def __init__(self, *args, **kwargs):
#         self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

#         super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

#         self.feat_conv = nn.Sequential(
#             nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
#         )

#         self.feat_ext = nn.Sequential(
#             nn.Conv2d(self.out_channels * 2, self.out_channels, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(self.out_channels, 2, 3, 1, 1)
#         )

#         self.init_offset()

#     def init_offset(self):

#         def _constant_init(module, val, bias=0):
#             if hasattr(module, 'weight') and module.weight is not None:
#                 nn.init.constant_(module.weight, val)
#             if hasattr(module, 'bias') and module.bias is not None:
#                 nn.init.constant_(module.bias, bias)

#         #_constant_init(self.conv_offset[-1], val=0, bias=0)

#     def forward(self, x, fea_cur, res, flow):
#         #feat_cur = self.feat_ext(frm_cur)
#         #res = frm_cur - flow_warp(frm_pre, flow)
#         res = abs(res)
#         B, C, H, W = res.shape
#         res = res.sum(dim=1)
#         res = F.sigmoid(res).view(B, 1, H, W)

#         fea_cur = self.feat_conv(fea_cur)
#         aligned_feat_pre = flow_warp(x, flow)
#         extra_feat = torch.cat([aligned_feat_pre, fea_cur], dim=1)
#         residual = self.feat_ext(extra_feat)
#         residual = residual.permute(0,2,3,1)
#         res = res.permute(0,2,3,1)

#         flow = flow + residual * res

#         aligned_feat_pre = flow_warp(x, flow)
#         return aligned_feat_pre


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)



def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


"""
FastDVDnet denoising algorithm

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import torch
import torch.nn.functional as F

def temp_denoise(model, noisyframe, sigma_noise):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%4
	if expanded_h:
		expanded_h = 4-expanded_h
	expanded_w = sh_im[-1]%4
	if expanded_w:
		expanded_w = 4-expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

	# denoise
	out = torch.clamp(model(noisyframe, sigma_noise), 0., 1.)

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out


def denoise_seq_fastdvdnet_batched(seq, noise_std, temp_psz, model_temporal):
    r"""Denoises a batched sequence of frames with FastDVDnet.

    Args:
        seq: Tensor. [B, T, C, H, W] input noisy frames
        noise_std: Tensor or float. shape: [B] or [1] or scalar std value
        temp_psz: int. temporal window size (e.g., 5)
        model_temporal: instance of FastDVDnet model
    Returns:
        denframes: Tensor. [B, T, C, H, W] denoised frames
    """
    B, T, C, H, W = seq.shape
    device = seq.device
    ctrlfr_idx = (temp_psz - 1) // 2

    # 生成噪声图 [B, 1, H, W]
    if isinstance(noise_std, float) or isinstance(noise_std, int):
        noise_map = torch.full((B, 1, H, W), noise_std, device=device)
    elif isinstance(noise_std, torch.Tensor):
        if noise_std.ndim == 1:
            noise_map = noise_std.view(B, 1, 1, 1).expand(B, 1, H, W)
        else:
            noise_map = noise_std.to(device)
    else:
        raise ValueError("Unsupported noise_std type.")

    # 初始化输出
    denframes = torch.empty((B, T, C, H, W), device=device)

    for t in range(T):
        indices = []
        for i in range(temp_psz):
            relidx = t + i - ctrlfr_idx
            relidx = min(max(relidx, 0), T - 1)  # 边界反射
            indices.append(relidx)

        # [B, temp_psz, C, H, W] → [B, temp_psz*C, H, W]
        inp = seq[:, indices, :, :, :]  # [B, temp_psz, C, H, W]
        inp = inp.view(B, temp_psz * C, H, W)

        # Padding 到 4 倍数
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        pad = (0, pad_w, 0, pad_h)

        inp = F.pad(inp, pad, mode='reflect')
        noise_map_p = F.pad(noise_map, pad, mode='reflect')

        # Denoise
        with torch.no_grad():
            out = torch.clamp(model_temporal(inp, noise_map_p), 0., 1.)

        if pad_h > 0:
            out = out[:, :, :-pad_h, :]
        if pad_w > 0:
            out = out[:, :, :, :-pad_w]

        denframes[:, t] = out

    return denframes


import torch.nn as nn
import torch
        
class CONTRIQUE_model(nn.Module):
    # resnet50 architecture with projector
    def __init__(self, args, encoder, n_features, \
                 patch_dim = (2,2), normalize = True, projection_dim = 128):
        super(CONTRIQUE_model, self).__init__()

        self.normalize = normalize
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.n_features = n_features
        self.patch_dim = patch_dim
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool_patch = nn.AdaptiveAvgPool2d(patch_dim)

        # MLP for projector
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )
        
    def forward(self, x_i, x_j):
        # global features
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        # local features
        h_i_patch = self.avgpool_patch(h_i)
        h_j_patch = self.avgpool_patch(h_j)
        
        h_i_patch = h_i_patch.reshape(-1,self.n_features,\
                                      self.patch_dim[0]*self.patch_dim[1])
        
        h_j_patch = h_j_patch.reshape(-1,self.n_features,\
                                      self.patch_dim[0]*self.patch_dim[1])
        
        h_i_patch = torch.transpose(h_i_patch,2,1)
        h_i_patch = h_i_patch.reshape(-1, self.n_features)
        
        h_j_patch = torch.transpose(h_j_patch,2,1)
        h_j_patch = h_j_patch.reshape(-1, self.n_features)
        
        h_i = self.avgpool(h_i)
        h_j = self.avgpool(h_j)
        
        h_i = h_i.view(-1, self.n_features)
        h_j = h_j.view(-1, self.n_features)
        
        if self.normalize:
            h_i = nn.functional.normalize(h_i, dim=1)
            h_j = nn.functional.normalize(h_j, dim=1)
            
            h_i_patch = nn.functional.normalize(h_i_patch, dim=1)
            h_j_patch = nn.functional.normalize(h_j_patch, dim=1)
        
        # global projections
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        # local projections
        z_i_patch = self.projector(h_i_patch)
        z_j_patch = self.projector(h_j_patch)
        
        return z_i, z_j, z_i_patch, z_j_patch, h_i, h_j, h_i_patch, h_j_patch