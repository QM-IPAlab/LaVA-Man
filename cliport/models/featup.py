import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.autograd import Function
import torch

import adaptive_conv_cuda_impl as cuda_impl
import adaptive_conv_cpp_impl as cpp_impl

torch.manual_seed(42)


class AdaptiveConv(Function):

    @staticmethod
    def forward(ctx, input, filters):
        ctx.save_for_backward(filters, input)
        b, h2, w2, f1, f2 = filters.shape
        assert f1 == f2

        if input.is_cuda:
            assert filters.is_cuda
            result = cuda_impl.forward(input, filters)
        else:
            result = cpp_impl.forward(input, filters)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        filters, input = ctx.saved_tensors
        grad_input = grad_filters = None
        b, h2, w2, f1, f2 = filters.shape
        assert f1 == f2

        grad_output = grad_output.contiguous()
        if grad_output.is_cuda:
            assert input.is_cuda
            assert filters.is_cuda
            if ctx.needs_input_grad[0]:
                grad_input = cuda_impl.grad_input(grad_output, filters)
            if ctx.needs_input_grad[1]:
                grad_filters = cuda_impl.grad_filters(grad_output, input)
        else:
            if ctx.needs_input_grad[0]:
                grad_input = cpp_impl.grad_input(grad_output, filters)
            if ctx.needs_input_grad[1]:
                grad_filters = cpp_impl.grad_filters(grad_output, input)

        return grad_input, grad_filters


class JBULearnedRange(torch.nn.Module):

    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=3):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))
        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim + self.diameter ** 2, self.diameter ** 2, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = self.range_proj(x)
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded) \
            .reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_x), dim=1)

    def get_spatial_kernel(self, device):
        dist_range = torch.linspace(-1, 1, self.diameter, device=device)
        x, y = torch.meshgrid(dist_range, dist_range)
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        spatial_kernel = self.get_spatial_kernel(source.device)
        range_kernel = self.get_range_kernel(guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel += .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1))
        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
        result = AdaptiveConv.apply(hr_source_padded, combined_kernel)
        return result


class JBUStack(torch.nn.Module):

    def __init__(self, feat_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up1 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up2 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up3 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up4 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=1))

    def upsample(self, source, guidance, up):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        upsampled = up(source, small_guidance)
        return upsampled

    def forward(self, source, guidance):
        source_2 = self.upsample(source, guidance, self.up1)
        source_4 = self.upsample(source_2, guidance, self.up2)
        source_8 = self.upsample(source_4, guidance, self.up3)
        source_16 = self.upsample(source_8, guidance, self.up4)
        return self.fixup_proj(source_16) * 0.1 + source_16
