from model_interface import GenerativeModel
from real_nvp_core.real_nvp_core import RealNVP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class RealNvpWrapper(nn.Module, GenerativeModel):
    def __init__(self, in_channels, input_shape, preprocess, n_scales, type, **kwargs):
        nn.Module.__init__(self)
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.real_nvp_core = RealNVP(in_channels=in_channels, preprocess=preprocess, n_scales=n_scales,
                                     device=self.device, type=type, input_shape=input_shape)
        self.prior = torch.distributions.Normal(torch.tensor(0.).to(self.device),
                                                torch.tensor(1.).to(self.device))
        self.input_shape = input_shape
        self.preprocess = preprocess

    def forward(self, x):
        z, log_det_J = self.real_nvp_core(x)
        return {'z': z,
                'log_det_J': log_det_J}

    def likelihood(self, input):
        with torch.no_grad():
            if isinstance(input, DataLoader):
                results = []
                for x, _ in input:
                    results.append(self.likelihood(x))
                results = np.concatenate(results, axis=0)
            else:
                z, log_det_J = self.real_nvp_core(input.to(self.device))
                log_p = torch.sum(self.prior.log_prob(z), dim=[1, 2, 3]) + log_det_J
                results = log_p.cpu().numpy()
        return results

    def sample(self, n):
        z = self.prior.sample(([n] + self.input_shape))
        x, _ = self.real_nvp_core(z.to(self.device), reverse=True)
        return x

    def criterion(self, z, log_det_J):
        if self.preprocess:
            log_det_J -= np.log(256) * np.product(self.input_shape)
        prior_ll = self.prior.log_prob(z)
        prior_ll_sum = torch.sum(prior_ll, dim=[1, 2, 3])
        log_p = prior_ll_sum + log_det_J
        return -torch.mean(log_p, dim=0)

    def get_mask(self, x, flag):
        add = 0 if flag else 1
        indices_sum = torch.stack(torch.meshgrid(torch.arange(x.shape[-2]), torch.arange(x.shape[-1]))).sum(axis=0)
        checkerboard = (indices_sum + add) % 2
        checkerboard = checkerboard.unsqueeze(dim=0)
        checkerboard = torch.repeat_interleave(checkerboard, x.shape[1], dim=0)
        checkerboard = checkerboard.unsqueeze(0)
        # checkerboard = torch.repeat_interleave(checkerboard, x.shape[0], dim=0)
        return checkerboard.to(self.device)
