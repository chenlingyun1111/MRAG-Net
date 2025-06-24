import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class DropBlock(nn.Module):
    def __init__(self, block_size=5, p=0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x):
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x):
        N, C, H, W = x.size()  # [4,64,288,288]
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super(SelfAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)  # x:[4,1024,18,18]
        x2, _ = torch.max(x, dim=1, keepdim=True) # x2:[4,1,18,18]
        x3 = torch.cat((x1, x2), dim=1)  # x3:[4,2,18,18]
        x4 = torch.sigmoid(self.conv(x3)) # x4:[4,1,18,18]
        x = x4 * x # x: [4,1024,18,18]
        assert len(x.shape) == 4
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            nn.BatchNorm2d(self.inter_channels),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        input_size = x.size()   # x:[4,1024,18,18]
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()  # 目标尺寸theta_x:[4,1024,18,18]
        phi_g = self.phi(g)  # 示例特征[4,1024,18,18]
        phi_g = F.interpolate(phi_g, size=theta_x_size[2:], mode='bilinear', align_corners=True)  # x:[4,1024,18,18]
        f = F.relu(theta_x + phi_g, inplace=True)

        psi_f = self.psi(f)

        return psi_f


class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(M_Conv, self).__init__()
        pad_size = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=pad_size, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x) # [4,3,288,288]
        return x   # [4,64,288,288]


def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

# 现了一个快速引导滤波器，其中融入了注意力机制。它通常用于图像处理任务，例如图像去噪、细节增强等
class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        mean_a = self.boxfilter(l_a) / N
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        mean_ay = self.boxfilter(l_a * lr_y) / N
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        mean_ax = self.boxfilter(l_a * lr_x) / N

        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A*hr_x+mean_b).float()


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class ConvNext(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        pad_size = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad_size, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_block = DropBlock(7, 0.5)

    def forward(self, x):
        _input = x  # [4,64,288,288]
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.conv1(x)  # [4,256,288,288]
        x = self.act(x)
        x = self.conv2(x) #[4,64,288,288]
        x = self.gamma.unsqueeze(-1).unsqueeze(-1) * x
        x = _input + self.drop_block(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class MRAGNet(nn.Module):
    def __init__(self, channel, n_classes, base_c, depths, kernel_size):
        super(MRAGNet, self).__init__()

        self.input_layer = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=kernel_size),
            *[ConvNext(base_c * 1, kernel_size=kernel_size) for _ in range(depths[0])]
        )
        self.input_skip = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=kernel_size),
        )
        self.conv1 = M_Conv(channel, base_c * 1, kernel_size=3)

        self.down_conv_2 = nn.Sequential(*[
            nn.Conv2d(base_c * 2, base_c * 2, kernel_size=2, stride=2),
            *[ConvNext(base_c * 2, kernel_size=kernel_size) for _ in range(depths[1])]
            ])
        self.conv2 = M_Conv(channel, base_c * 2, kernel_size=3)

        self.down_conv_3 = nn.Sequential(*[
            nn.Conv2d(base_c * 4, base_c * 4, kernel_size=2, stride=2),
            *[ConvNext(base_c * 4, kernel_size=kernel_size) for _ in range(depths[2])]
            ])
        self.conv3 = M_Conv(channel, base_c * 4, kernel_size=3)

        self.down_conv_4 = nn.Sequential(*[
            nn.Conv2d(base_c * 8, base_c * 8, kernel_size=2, stride=2),
            *[ConvNext(base_c * 8, kernel_size=kernel_size) for _ in range(depths[3])]
        ])
        self.conv4 = M_Conv(channel, base_c * 8, kernel_size=3)

        self.down_conv_5 = nn.Sequential(*[
            nn.Conv2d(base_c * 16, base_c * 16, kernel_size=2, stride=2),
            *[ConvNext(base_c * 16, kernel_size=kernel_size) for _ in range(depths[4])]
            ])
        self.attn = SelfAttentionBlock()

        self.up_residual_conv4 = ResidualConv(base_c * 16, base_c * 8, 1, 1)
        self.up_residual_conv3 = ResidualConv(base_c * 8, base_c * 4, 1, 1)
        self.up_residual_conv2 = ResidualConv(base_c * 4, base_c * 2, 1, 1)
        self.up_residual_conv1 = ResidualConv(base_c * 2, base_c * 1, 1, 1)

        self.output_layer4 = nn.Sequential(
            nn.Conv2d(base_c * 8, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer3 = nn.Sequential(
            nn.Conv2d(base_c * 4, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer2 = nn.Sequential(
            nn.Conv2d(base_c * 2, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer1 = nn.Sequential(
            nn.Conv2d(base_c * 1, n_classes, 1, 1),
            nn.Sigmoid(),
        )

        self.fgf = FastGuidedFilter_attention(r=2, eps=1e-2)
        self.attention_block4 = CrossAttentionBlock(in_channels=base_c * 16)
        self.attention_block3 = CrossAttentionBlock(in_channels=base_c * 8)
        self.attention_block2 = CrossAttentionBlock(in_channels=base_c * 4)
        self.attention_block1 = CrossAttentionBlock(in_channels=base_c * 2)

        self.conv_cat_4 = M_Conv(base_c * 16 + base_c * 16, base_c * 16, kernel_size=1)
        self.conv_cat_3 = M_Conv(base_c * 8 + base_c * 8, base_c * 8, kernel_size=1)
        self.conv_cat_2 = M_Conv(base_c * 8 + base_c * 4, base_c * 4, kernel_size=1)
        self.conv_cat_1 = M_Conv(base_c * 4 + base_c * 2, base_c * 2, kernel_size=1)

    def forward(self, x):
        # Get multi-scale from input
        _, _, h, w = x.size()  # [4,3,288,288]

        x_scale_2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True) # [4,3,144,144]
        x_scale_3 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True) #[4,3,72,72]
        x_scale_4 = F.interpolate(x, size=(h // 8, w // 8), mode='bilinear', align_corners=True)# [4,3,36,36]

        # Encoder
        x1 = self.input_layer(x) + self.input_skip(x)  # 初始特征提取。
        x1_conv = self.conv1(x)  # 原始输入提取特征
        x1_down = torch.cat([x1_conv, x1], dim=1)  # 输出拼接

        x2 = self.down_conv_2(x1_down)  # 第二层下采样和特征提取
        x2_conv = self.conv2(x_scale_2)  # 从 1/2 尺度输入提取的特征。
        x2_down = torch.cat([x2_conv, x2], dim=1)  # 拼接

        x3 = self.down_conv_3(x2_down)   # 第三层下采样和特征提取。
        x3_conv = self.conv3(x_scale_3)  # 从 1/4 尺度输入提取的特征
        x3_down = torch.cat([x3_conv, x3], dim=1)

        x4 = self.down_conv_4(x3_down)  # 第三层下采样和特征提取。
        x4_conv = self.conv4(x_scale_4)  # 从 1/4 尺度输入提取的特征 x4_conv:[4,512,36,36]
        x4_down = torch.cat([x4_conv, x4], dim=1) # x4:[4,1028,36,36]

        x5 = self.down_conv_5(x4_down) #x5:[4,1024,18,18]
        x5 = self.attn(x5)  # 在最深层特征上应用自注意力。

        # Decoder
        _, _, h, w = x4_down.size()
        x4_gf = torch.cat([x4_down, F.interpolate(x5, size=(h, w), mode='bilinear', align_corners=True)], dim=1)  # 将上采样后的深层特征与来自编码器路径的对应尺度特征进行拼接。
        x4_gf_conv = self.conv_cat_4(x4_gf)  # 调整拼接后的通道数
        x4_small = F.interpolate(x4_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)  # 当前尺度的特征。
        fgf_out = self.fgf(x4_small, x5, x4_gf_conv, self.attention_block4(x4_small, x5))
        x4_up = self.up_residual_conv4(fgf_out)  # 上一层解码器上采样后的特征。经过引导滤波后的特征，再通过残差卷积进行上采样。

        _, _, h, w = x3_down.size()
        x3_gf = torch.cat([x3_down, F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)], dim=1) # 将上采样后的深层特征与来自编码器路径的对应尺度特征进行拼接。
        x3_gf_conv = self.conv_cat_3(x3_gf) # 调整拼接后的通道数
        x3_small = F.interpolate(x3_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)  # 当前尺度的特征。
        fgf_out = self.fgf(x3_small, x4, x3_gf_conv, self.attention_block3(x3_small, x4_up))
        x3_up = self.up_residual_conv3(fgf_out)  # 上一层解码器上采样后的特征。经过引导滤波后的特征，再通过残差卷积进行上采样。

        _, _, h, w = x2_down.size()
        x2_gf = torch.cat([x2_down, F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x2_gf_conv = self.conv_cat_2(x2_gf)
        x2_small = F.interpolate(x2_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x2_small, x3_up, x2_gf_conv, self.attention_block2(x2_small, x3_up))
        x2_up = self.up_residual_conv2(fgf_out)

        _, _, h, w = x1_down.size()
        x1_gf = torch.cat([x1_down, F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x1_gf_conv = self.conv_cat_1(x1_gf)
        x1_small = F.interpolate(x1_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x1_small, x2_up, x1_gf_conv, self.attention_block1(x1_small, x2_up))
        x1_up = self.up_residual_conv1(fgf_out)

        _, _, h, w = x.size()
        out_4 = F.interpolate(x4_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True) # 将不同尺度的输出上采样到原始输入图像的尺寸。
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_4 = self.output_layer3(out_3)
        out_3 = self.output_layer3(out_3)  # 最终的 1x1 卷积和 Sigmoid 激活，生成分割结果。
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        return out_1, out_2, out_3, out_4
