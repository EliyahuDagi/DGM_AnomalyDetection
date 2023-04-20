import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from real_nvp_core.coupling_layer import CheckerBoardCoupling, ChannelWiseCoupling


class RealNvpScale(nn.Module):
    def __init__(self, in_channels, device, type):
        nn.Module.__init__(self)
        self.device = device
        self.in_channels = in_channels
        if type == 'original':
            mid_channels = 8 * in_channels
        else:
            mid_channels = 2 * in_channels
        self.checkerboard_couplings = nn.ModuleList([
            CheckerBoardCoupling(in_channels, mid_channels, out_channels=in_channels * 2, switch=False, device=self.device),
            CheckerBoardCoupling(in_channels, mid_channels, out_channels=in_channels * 2, switch=True, device=self.device),
            CheckerBoardCoupling(in_channels, mid_channels, out_channels=in_channels * 2, switch=False, device=self.device),
        ])
        self.channel_wise_couplings = nn.ModuleList([
            ChannelWiseCoupling(4 * in_channels, 8, out_channels=4 * in_channels, switch=False, device=self.device),
            ChannelWiseCoupling(4 * in_channels, 8, out_channels=4 * in_channels, switch=True, device=self.device),
            ChannelWiseCoupling(4 * in_channels, 8, out_channels=4 * in_channels, switch=False, device=self.device)
        ])

    def forward(self, x, j_det=None, reverse=False):
        if reverse:
            x = squeeze_2x2(x, reverse=False)
            for coupling in reversed(self.channel_wise_couplings):
                x, j_det = coupling(x, j_det, True)
            x = squeeze_2x2(x, reverse=True)
            for coupling in reversed(self.checkerboard_couplings):
                x, j_det = coupling(x, j_det, True)
        else:
            # Checker Board --> Squeeze -->  ChannelWise --> un squeeze
            for coupling in self.checkerboard_couplings:
                x, j_det = coupling(x, j_det, False)
            x = squeeze_2x2(x, reverse=False)
            for coupling in self.channel_wise_couplings:
                x, j_det = coupling(x, j_det, False)
            x = squeeze_2x2(x, reverse=True)
        return x, j_det


class RealNVP(nn.Module):
    def __init__(self, in_channels, preprocess, n_scales, device, type, input_shape):
        super(RealNVP, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.register_buffer('alpha', torch.tensor([1e-5], dtype=torch.float32, device=self.device))
        self.input_shape = input_shape
        self.n_scales = n_scales
        # before each scale we perform squeeze that doubles the number of channel so
        # input channels in each scale will be 2^scale
        # since when modeling features we will have large number of input channels
        # and multiply also by 2^scale, it is problam in a manner of computation resource
        # so in features ive used smaller number of mid-level features
        self.scales = nn.ModuleList([RealNvpScale(in_channels * (2 ** scale), device, type=type)
                                     for scale in range(n_scales)])
        # final coupling
        self.checkerboard_couplings2 = nn.ModuleList([
            CheckerBoardCoupling(in_channels * 2, 32, out_channels=in_channels * 4, switch=True, device=self.device),
            CheckerBoardCoupling(in_channels * 2, 32, out_channels=in_channels * 4, switch=False, device=self.device),
            CheckerBoardCoupling(in_channels * 2, 32, out_channels=in_channels * 4, switch=True, device=self.device),
        ])
        self.channel_wise_couplings2 = nn.ModuleList([
            ChannelWiseCoupling(8 * in_channels, 32, out_channels=8 * in_channels, switch=True, device=self.device),
            ChannelWiseCoupling(8 * in_channels, 32, out_channels=8 * in_channels, switch=False, device=self.device),
            ChannelWiseCoupling(8 * in_channels, 32, out_channels=8 * in_channels, switch=True, device=self.device)
        ])
        self.checkerboard_last_couplings = nn.ModuleList([
            CheckerBoardCoupling(in_channels, 8, 2 * in_channels, switch=False, device=self.device),
            CheckerBoardCoupling(in_channels, 8, 2 * in_channels, switch=True, device=self.device),
        ])
        self.preprocess = preprocess

    def forward(self, x, reverse=False):
        j_det = None
        if reverse:
            j_det = torch.zeros(x.shape[0])
            for coupling in reversed(self.checkerboard_last_couplings):
                x, j_det = coupling(x, j_det, True)
            # reverse build of scales tensors shapes
            scales_splits = []
            for i in range(self.n_scales):
                # x, j_det = scale(x, j_det, reverse=False)
                if i < self.n_scales - 1:
                    x = squeeze_2x2(x, reverse=False, alt_order=True)
                    prev_scale, x = x.chunk(2, dim=1)
                    scales_splits.append(prev_scale)
                else:
                    scales_splits.append(x)
            # Reverse from gaussian to image
            for i, scale in reversed(list(enumerate(self.scales))):
                x, j_det = scale(scales_splits[i], j_det, reverse=True)
                if i > 0:
                    prev_input = torch.cat([scales_splits[i-1], x], dim=1)
                    prev_input = squeeze_2x2(prev_input, reverse=True, alt_order=True)
                    scales_splits[i - 1] = prev_input

            x = torch.sigmoid(x)
            x = (x - 0.5 * self.alpha) / (1 - self.alpha)
        else:

            if self.preprocess:
                # Dequantization and logits conversion
                x, j_det = self._pre_process(x)
            else:
                j_det = torch.zeros(x.shape[0]).to(x.device)

            scales_res = []
            for i, scale in enumerate(self.scales):
                # apply cur scale then squeeze and apply the next scale on half of the channels
                x, j_det = scale(x, j_det, reverse=False)
                if i < self.n_scales - 1:
                    x = squeeze_2x2(x, reverse=False, alt_order=True)
                    cur_scale, x = x.chunk(2, dim=1)
                    scales_res.append(cur_scale)
                else:
                    scales_res.append(x)
            # concat scales results bottom till up
            for i in range(len(scales_res) - 1, 0, -1):
                combind_scales = torch.cat([scales_res[i-1], scales_res[i]], dim=1)
                combind_scales = squeeze_2x2(combind_scales, reverse=True, alt_order=True)
                scales_res[i - 1] = combind_scales
            x = scales_res[0]

            for coupling in self.checkerboard_last_couplings:
                x, j_det = coupling(x, j_det, False)
        return x, j_det

    def _pre_process(self, x):
        # DeQuantization
        y = (x * 255. + torch.rand_like(x)) / 256.
        # model the inverse logit
        y = y * (1 - self.alpha) + self.alpha * 0.5
        log_det_j = torch.sum(-torch.log(y) - torch.log(1 - y), dim=[1, 2, 3])
        log_det_j -= np.log(256) * np.product(self.input_shape)
        y = torch.log(y) - torch.log(1. - y)
        return y, log_det_j


# I've taken this function as is from existing repository
def squeeze_2x2(x, reverse=False, alt_order=False):
    """For each spatial position, a sub-volume of shape `1x1x(N^2 * C)`,
    reshape into a sub-volume of shape `NxNxC`, where `N = block_size`.

    Adapted from:
        https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_utils.py

    See Also:
        - TensorFlow nn.depth_to_space: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        reverse (bool): Whether to do a reverse squeeze (unsqueeze).
        alt_order (bool): Whether to use alternate ordering.
    """
    block_size = 2
    if alt_order:
        n, c, h, w = x.size()

        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels must be divisible by 4, got {}.'.format(c))
            c //= 4
        else:
            if h % 2 != 0:
                raise ValueError('Height must be divisible by 2, got {}.'.format(h))
            if w % 2 != 0:
                raise ValueError('Width must be divisible by 4, got {}.'.format(w))
        # Defines permutation of input channels (shape is (4, 1, 2, 2)).
        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
        perm_weight = torch.zeros((4 * c, c, 2, 2), dtype=x.dtype, device=x.device)
        for c_idx in range(c):
            slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
        shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)]
                                        + [c_idx * 4 + 1 for c_idx in range(c)]
                                        + [c_idx * 4 + 2 for c_idx in range(c)]
                                        + [c_idx * 4 + 3 for c_idx in range(c)],
                                        device=x.device)
        perm_weight = perm_weight[shuffle_channels, :, :, :]

        if reverse:
            x = F.conv_transpose2d(x, perm_weight, stride=2)
        else:
            x = F.conv2d(x, perm_weight, stride=2)
    else:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)

        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels {} is not divisible by 4'.format(c))
            x = x.view(b, h, w, c // 4, 2, 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.contiguous().view(b, 2 * h, 2 * w, c // 4)
        else:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError('Expected even spatial dims HxW, got {}x{}'.format(h, w))
            x = x.view(b, h // 2, 2, w // 2, 2, c)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, h // 2, w // 2, c * 4)

        x = x.permute(0, 3, 1, 2)

    return x