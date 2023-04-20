import numpy as np
import torch
import torch.nn as nn
from model_interface import GenerativeModel
from torch.utils.data import DataLoader


class ResNet(nn.Module):
    def __init__(self, in_channels, feat_channels, out_channels):
        super(ResNet, self).__init__()
        self.nn = base_nn(in_channels, feat_channels, out_channels)
        self.conv_in = None
        if in_channels == out_channels:
            self.conv_in = nn.Identity()
        else:
            self.conv_in = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.nn(x) + self.conv_in(x)


def base_nn(in_channels, feat_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, feat_channels, 3, 1, 1),
        nn.BatchNorm2d(feat_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(feat_channels, feat_channels, 3, 1, 1),
        nn.BatchNorm2d(feat_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(feat_channels, out_channels, 1, 1, 0),
    )


class VAE(nn.Module, GenerativeModel):
    def __init__(self, in_channels, latent_dim, input_shape, type):
        """

        :param in_channels: number of input channels 1 - grayscale 3 - color or arbitrary size
        :param latent_dim: dimension of mu and log var
        :param input_shape: the shape of the input
        :param type: original/features
        """
        nn.Module.__init__(self)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.feat_channels = 16
        self.img_size = input_shape[-1]
        self.in_norm = nn.Identity() if type == 'original' else nn.Sigmoid()
        # sequence of residual block + max pooling
        self.encoder = nn.Sequential(ResNet(in_channels, 32, 32),
                                     nn.MaxPool2d(2),  # B,  32, 16, 16
                                     ResNet(32, 32, 32),
                                     nn.MaxPool2d(2),  # B, 32, 8, 8
                                     ResNet(32, 64, self.feat_channels),
                                     nn.MaxPool2d(2),  # B,  32,  4, 4
                                     )
        self.emd_spatial = int(self.img_size / 8)
        full_spatial_dim = self.emd_spatial ** 2
        self.mu = nn.Linear(self.feat_channels * full_spatial_dim, latent_dim)
        self.logvar = nn.Linear(self.feat_channels * full_spatial_dim, latent_dim)

        self.upsample = nn.Linear(latent_dim, self.feat_channels * self.emd_spatial * self.emd_spatial)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feat_channels, self.feat_channels * 2, 2, 2),  # B,  16,  8,  8
            nn.BatchNorm2d(self.feat_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(self.feat_channels * 2, self.feat_channels, 2, 2),  # B,  8, 16, 16
            nn.BatchNorm2d(self.feat_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(self.feat_channels, in_channels, 2, 2),  # B, 1, 32, 32
            nn.Sigmoid(),
        )
        self.bce = nn.BCELoss(reduction='none')

    def sample(self, sample_size, mu=None, log_var=None):
        """
        param sample_size: Number of samples
        param mu: z mean, None for prior (init with zeros)
        param logvar: z logstd, None for prior (init with zeros)
        return:
        """
        if mu is None:
            mu = torch.zeros((sample_size, self.latent_dim)).to(self.device)
        if log_var is None:
            log_var = torch.zeros((sample_size, self.latent_dim)).to(self.device)
        z_sample = self.z_sample(mu=mu, log_var=log_var)
        return self.decode(z_sample)

    def z_sample(self, mu, log_var):
        # re parametrization trick
        eps = torch.randn_like(mu).to(self.device)
        return mu + torch.exp(log_var * 0.5) * eps

    def ELBO(self, x, reconstruction, mu, log_var):
        # calculate reconstruction and kl divergence term
        reconstruct_term = torch.sum(self.bce(reconstruction, self.in_norm(x)), dim=[1, 2, 3])
        kl_term = torch.sum(-0.5 * (1 + log_var - (mu * mu) - torch.exp(log_var)), dim=1)
        return -reconstruct_term - kl_term

    def criterion(self, x, reconstruction, mu, log_var):
        elbo = self.ELBO(x=x, reconstruction=reconstruction, mu=mu, log_var=log_var)
        return torch.mean(-elbo)

    # TODO use importance weight to evaluate likelihood
    def likelihood(self, input):
        with torch.no_grad():
            if isinstance(input, DataLoader):
                results = []
                for x, _ in input:
                    results.append(self.likelihood(x.to(self.device)))
                results = np.concatenate(results, axis=0)
            else:
                results = self.ELBO(**self.forward(input.to(self.device))).cpu().numpy()
        return results

    def decode(self, z_sample):
        z = self.upsample(z_sample)
        z = z.reshape(z_sample.shape[0], self.feat_channels, self.emd_spatial, self.emd_spatial)
        return self.decoder(z)

    def forward(self, x):
        encoded = nn.Flatten()(self.encoder(self.in_norm(x)))
        mu = self.mu(encoded)
        log_var = self.logvar(encoded)
        # sample from q(z|x)
        z_sample = self.z_sample(mu=mu, log_var=log_var)
        reconstruction = self.decode(z_sample)
        return {'x': x,
                'mu': mu,
                'log_var': log_var,
                'reconstruction': reconstruction
                }
