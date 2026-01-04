"""
Definition of the FastDVDnet model
"""
import torch
import torch.nn as nn

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(3+1), num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class DenBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(DenBlock, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 - x

		return x

class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.temp1 = DenBlock(num_input_frames=3)
		self.temp2 = DenBlock(num_input_frames=3)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1, x2, x3, x4) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))

		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map)
		x21 = self.temp1(x1, x2, x3, noise_map)
		x22 = self.temp1(x2, x3, x4, noise_map)

		#Second stage
		x = self.temp2(x20, x21, x22, noise_map)

		return x



import torch
import torch.nn as nn
import torch.nn.functional as F

class FastDVDnetWrapper(nn.Module):
    """
    封装 FastDVDnet 模型，使其支持 [B, T, C, H, W] 输入和输出。
    自动处理时序窗口、padding、噪声图。
    """
    def __init__(self, fastdvdnet_model: nn.Module, temp_psz=5, noise_std=25 / 255.):
        super().__init__()
        self.model = fastdvdnet_model
        self.temp_psz = temp_psz
        self.noise_std = noise_std

        # sanity check
        assert temp_psz == 5, "当前 FastDVDnet 实现只支持 temp_psz=5（即五帧推理）"

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, C, H, W] in range [0., 1.]
        Returns:
            output: [B, T, C, H, W] denoised video
        """
        B, T, C, H, W = x.shape
        device = x.device
        ctrlfr_idx = self.temp_psz // 2  # index of center frame

        # 构建噪声图 [B, 1, H, W]
        noise_map = torch.full((B, 1, H, W), self.noise_std, device=device)

        # 初始化输出
        out = torch.empty((B, T, C, H, W), device=device)

        for t in range(T):
            indices = []
            for i in range(self.temp_psz):
                rel_t = t + i - ctrlfr_idx
                rel_t = max(0, min(T - 1, rel_t))  # 边界反射
                indices.append(rel_t)

            clip = x[:, indices, :, :, :]  # [B, 5, C, H, W]
            clip = clip.reshape(B, self.temp_psz * C, H, W)  # [B, 15, H, W]

            # Pad 到4倍数
            pad_h = (4 - H % 4) % 4
            pad_w = (4 - W % 4) % 4
            if pad_h or pad_w:
                clip = F.pad(clip, (0, pad_w, 0, pad_h), mode="reflect")
                noise_pad = F.pad(noise_map, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                noise_pad = noise_map

            # with torch.no_grad():
            #     pred = self.model(clip, noise_pad)  # [B, C, H, W]
            #     pred = torch.clamp(pred, 0., 1.)
            pred = self.model(clip, noise_pad)  # [B, C, H, W]
            pred = torch.clamp(pred, 0., 1.)

            if pad_h:
                pred = pred[:, :, :-pad_h, :]
            if pad_w:
                pred = pred[:, :, :, :-pad_w]

            out[:, t] = pred

        return out