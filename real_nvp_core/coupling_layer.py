import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, in_channels, feat_channels, out_channels):
        super(ResNet, self).__init__()
        self.nn = base_nn(in_channels, feat_channels, out_channels)
        self.conv_in = None
        if in_channels == out_channels:
            self.conv_in = nn.Identity
        else:
            self.conv_in = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        return self.nn(x) + self.conv_in(x)


def base_nn(in_channels, feat_channels, out_channels):
    return nn.Sequential(
        nn.utils.weight_norm(nn.Conv2d(in_channels, feat_channels, 3, 1, 1)),
        nn.BatchNorm2d(feat_channels),
        nn.LeakyReLU(inplace=True),
        nn.utils.weight_norm(nn.Conv2d(feat_channels, feat_channels, 3, 1, 1)),
        nn.BatchNorm2d(feat_channels),
        nn.LeakyReLU(inplace=True),
        nn.utils.weight_norm(nn.Conv2d(feat_channels, feat_channels, 3, 1, 1)),
        nn.BatchNorm2d(feat_channels),
        nn.LeakyReLU(inplace=True),
        nn.utils.weight_norm(nn.Conv2d(feat_channels, out_channels, 1, 1, 0)),
    )


class CheckerBoardCoupling(nn.Module):

    def __init__(self, in_channels, features_channels, out_channels, switch, device):
        """

        @param in_channels: number of input channels
        @param features_channels: number of mid level channels
        @param out_channels: number of output channels
        @param switch: switch the checkerboard mask(on will be off and the opposite)
        @param device: which device to work with
        """
        nn.Module.__init__(self)
        self.device = device
        self.net = base_nn(in_channels=in_channels, feat_channels=features_channels, out_channels=out_channels)
        self.switch = switch
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def get_mask(self, x):
        add = 0 if self.switch else 1
        indices_sum = torch.stack(torch.meshgrid(torch.arange(x.shape[-2]), torch.arange(x.shape[-1]))).sum(axis=0)
        checkerboard = (indices_sum + add) % 2
        checkerboard = checkerboard.unsqueeze(dim=0)
        return checkerboard.to(self.device)

    def forward(self, x, j_det=None, reverse=True):
        # first create a checkerboard mask
        active_mask = self.get_mask(x)
        non_active_mask = 1 - active_mask
        # use the active part to calculate affine transform( log s, b )
        # apply it on the non-active part
        # reverse=False e^s * x[non-active] + t
        # reverse=True e^(-s) * (x[non-active] - t)
        # The log determinant of the Jacobian will be the sum of s
        z = x
        z_active = active_mask * z
        s, t = self.net(z_active).chunk(2, dim=1)
        s = self.rescale(torch.tanh(s))
        s = s * non_active_mask
        t = t * non_active_mask
        if reverse:
            z = (z - t) * torch.exp(-s)
        else:
            z = (torch.exp(s) * z) + t
            j_det += s.sum(dim=[1, 2, 3])
        return z, j_det


class ChannelWiseCoupling(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, switch, device):
        """

        @param in_channels: number of input channels
        @param mid_channel: number of mid level channels
        @param out_channels: number of output channels
        @param switch: switch the channels(on will be first half and the opposite)
        @param device: which device to work with
        """
        nn.Module.__init__(self)
        self.device = device
        self.net = base_nn(in_channels // 2, mid_channel, out_channels=out_channels)
        self.rescale = nn.utils.weight_norm(Rescale(in_channels // 2))
        self.switch = switch

    def forward(self, x, j_det, reverse=False):
        # split the channels to 2 parts
        if self.switch:
            x_active, x_inactive = x.chunk(2, dim=1)
        else:
            x_inactive, x_active = x.chunk(2, dim=1)
        # calculate affinity
        s, t = self.net(x_active).chunk(2, dim=1)
        s = self.rescale(torch.tanh(s))

        # Scale and translate
        # The log determinant of the Jacobian will be the sum of s
        if reverse:
            x_inactive = (x_inactive - t) * torch.exp(-s)
        else:
            x_inactive = (torch.exp(s) * x_inactive) + t
            j_det += torch.sum(s, dim=[1, 2, 3])

        if self.switch:
            x = torch.cat((x_active, x_inactive), dim=1)
        else:
            x = torch.cat((x_inactive, x_active), dim=1)
        return x, j_det


class Rescale(nn.Module):
    """
        Rescale each channel after TanH
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x

