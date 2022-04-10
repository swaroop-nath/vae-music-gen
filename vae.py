import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d
import numpy as np
from torch.nn.functional import leaky_relu, relu
from utils import log_p_x, kl_divergence, empirical_kl_divergence
from manager import DEVICE

class VAE(nn.Module):
    def __init__(self, num_instruments, temporal_unit_size, num_pitches, latent_dim=10, filter_size=5, **kwargs):
        super(VAE, self).__init__()
        amt_padding = 0
        self.latent_dim = latent_dim

        # Encoder
        # Input is of size - (batch_size, num_instruments, temporal_unit_size, num_pitches), width == temporal_unit_size, height == num_pitches
        self.conv1 = Conv2d(in_channels=num_instruments, out_channels=kwargs['enc-out-channel-1'], \
                        kernel_size=(4, 4), stride=(4, 4))
        width, height = self._get_transformed_width_height(temporal_unit_size, num_pitches, amt_padding, kwargs['enc-stride-1'], filter_width=4, filter_height=4)

        self.conv2 = Conv2d(in_channels=kwargs['enc-out-channel-1'], out_channels=kwargs['enc-out-channel-2'], \
                        kernel_size=(4, 4), stride=(4, 4))
        width, height = self._get_transformed_width_height(width, height, amt_padding, kwargs['enc-stride-2'], filter_width=4, filter_height=4)

        self.conv3 = Conv2d(in_channels=kwargs['enc-out-channel-2'], out_channels=kwargs['enc-out-channel-3'], \
                        kernel_size=(2, 8), stride=(2, 8))
        width, height = self._get_transformed_width_height(width, height, amt_padding, kwargs['enc-stride-3'], filter_width=2, filter_height=8)

        in_feats = int(width) * int(height) * kwargs['enc-out-channel-3']
        out_feats = 2 * latent_dim
        self.phi_fc = nn.Linear(in_features=in_feats, out_features=out_feats)

        # Decoder
        # Input is a sampled z, of size - (batch_size, temporal_unit_size, latent_dim)
        out_feats = kwargs['enc-out-channel-3']
        self.phi_fc_inv = nn.Linear(in_features=latent_dim, out_features=out_feats)
        
        self.de_conv1 = ConvTranspose2d(in_channels=kwargs['enc-out-channel-3'], out_channels=kwargs['enc-out-channel-2'], \
                        kernel_size=(2, 8), stride=(2, 8))

        self.de_conv2 = ConvTranspose2d(in_channels=kwargs['enc-out-channel-2'], out_channels=kwargs['enc-out-channel-1'], \
                        kernel_size=(4, 4), stride=(4, 4))

        self.de_conv3 = ConvTranspose2d(in_channels=kwargs['enc-out-channel-1'], out_channels=num_instruments, \
                        kernel_size=(4, 4), stride=(4, 4))

        # Define a special extra parameter to learn scalar sig_x for all pixels
        self.log_sig_x = nn.Parameter(torch.zeros(()))

    def encode(self, input_notes):
        input_notes = input_notes.unsqueeze(1)
        conv1_output = relu(self.conv1(input_notes))
        conv2_output = relu(self.conv2(conv1_output))
        conv3_output = relu(self.conv3(conv2_output))

        zs = conv3_output.view(input_notes.size(0), -1) # flattening
        mu_sigma = self.phi_fc(zs) # (batch_size, 2 * latent_dim)

        return mu_sigma[:, :self.latent_dim], torch.exp(mu_sigma[:, self.latent_dim:])

    def decode(self, sampled_z):
        # sampled_z is of size - (batch_size, temporal_unit_size, latent_dim)
        # Generation of size - (batch_size, temporal_unit_size, latent_dim)
        bsz, num_gen, _ = sampled_z.size()

        fc_inv_output = relu(self.phi_fc_inv(sampled_z)) # (batch_size, temporal_unit_size, num_patches) -> (batch_size, temporal_unit_size, computed width height)
        fc_inv_output = fc_inv_output.reshape(-1, fc_inv_output.size(-1))
        fc_inv_output = fc_inv_output.unsqueeze(-1).unsqueeze(-1) # (..., context width height, 1, 1)

        de_conv1_output = relu(self.de_conv1(fc_inv_output))
        de_conv2_output = relu(self.de_conv2(de_conv1_output))
        de_conv3_output = self.de_conv3(de_conv2_output) # (..., 1, width, height)

        de_conv3_output = de_conv3_output.squeeze(1)

        return de_conv3_output

    def forward(self, input):
        mu, sigma = self.encode(input)
        sampled_z = self._sample_z(mu, sigma)
        generated_notes = self.decode(sampled_z)

        ELBO = log_p_x(input, generated_notes, self.log_sig_x.exp()) - empirical_kl_divergence(sampled_z, mu, sigma)

        return generated_notes, -ELBO.unsqueeze(-1)

    def generate_music_roll(self, sample):
        sample = sample.unsqueeze(0)
        mu, sigma = self.encode(sample)
        sampled_z = self._sample_z(mu, sigma)
        gen_notes = self.decode(sampled_z)

        return gen_notes.squeeze(0)

    def _get_transformed_width_height(self, width, height, padding, stride, filter_width, filter_height):
        new_width = np.floor((width - filter_width + 2 * padding) / stride) + 1
        new_height = np.floor((height - filter_height + 2 * padding) / stride) + 1

        return new_width, new_height

    def _sample_z(self, mu, sigma, n_samples=1):
        eps = torch.randn(mu.size(0), n_samples).to(DEVICE)
        sampled_z = mu + eps * sigma
        sampled_z = sampled_z.unsqueeze(1)
        return sampled_z