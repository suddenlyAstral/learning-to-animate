import logging

import numpy as np
import torch
import torch.nn as nn

from wandb_utils import tensor2np

"""
The code is taken with heavy modification (VQVAE to CVAE) from here https://github.com/rosinality/vq-vae-2-pytorch
"""


class PrintLayer(nn.Module):
    """
    A transparent layer that prints tensor shape, for debugging purposes.
    """
    def __init__(self):
        super(PrintLayer, self).__init__()
        self.logger = logging.getLogger(__name__)

    def forward(self, x):
        self.logger.debug(str(x.size()))
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Model(nn.Module):
    def __init__(self,
                 in_channel=3,
                 channel=128,
                 n_res_block=2,
                 n_res_channel=32,
                 latent_dim=64,
                 use_time=True,
                 device="cpu"):
        """
        Initialize a hierarchical (C)VAE with 2 encoders/decoders.

        Args:
            in_channel - input image channels.
            channel - channels between bottom and top.
            n_res_block - number res blocks in each encoder/decoder.
            n_res_channel - number of channels in res blocks.
            latent_dim - number of channels in the latent code (of each of two encoder levels).
            use_time - whether to condition decoder on time. if False, this is just a VAE
            device - device
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.use_time = use_time

        # model parts that define the hierarchical encoder
        # input: image
        # output: 4 vectors of size latent_dim: (top_mu, top_logvar, bottom_mu, bottom_logvar)
        self.enc_bottom = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_top = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.mu_top = nn.Conv2d(channel, latent_dim, kernel_size=1)
        self.logvar_top = nn.Conv2d(channel, latent_dim, kernel_size=1)
        self.dec_internal = Decoder(latent_dim, latent_dim, channel, n_res_block, n_res_channel, stride=2)
        self.mu_bottom = nn.Conv2d(channel + latent_dim, latent_dim, kernel_size=1)
        self.logvar_bottom = nn.Conv2d(channel + latent_dim, latent_dim, kernel_size=1)

        dim_if_use_t = 1 if use_time else 0

        # model parts that define the hierarchical decoder
        self.upsample_top = nn.ConvTranspose2d(latent_dim + dim_if_use_t, latent_dim, 4, stride=2, padding=1)
        self.final_dec = Decoder(
            latent_dim + latent_dim + dim_if_use_t,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.to(device)

    def apply_encoder(self, x):
        """
        :param x: batch of images, tensor (B, C=3, H=256, W=256)
        :return: tuple (mu, logvar). each one is a tuple of two tensors:
                - (B, self.latent_dim, bottom_hw, bottom_hw)
                - (B, self.latent_dim, top_hw, top_hw)
        """
        code_bottom = self.enc_bottom(x)
        code_top = self.enc_top(code_bottom)

        mu_top = self.mu_top(code_top)
        logvar_top = self.logvar_top(code_top)

        z_internal = self.z_sample(mu_top, logvar_top)
        dec_internal = self.dec_internal(z_internal)
        code_bottom = torch.cat([dec_internal, code_bottom], dim=1)

        mu_bottom = self.mu_bottom(code_bottom)
        logvar_bottom = self.logvar_bottom(code_bottom)

        full_mu = (mu_bottom, mu_top)
        full_logvar = (logvar_bottom, logvar_top)
        return full_mu, full_logvar

    def apply_decoder(self, z):
        """
        :param z: tuple (z_bottom, z_top). tensors with sizes:
                - (B, self.latent_dim(+1), bottom_hw, bottom_hw)
                - (B, self.latent_dim(+1), top_hw, top_hw)
                where +1 is if we condition on time
        :return: tensor of generated images, (B, 3, 256, 256), in range [-1, 1]
        """
        z_bottom, z_top = z
        x_hat = self.upsample_top(z_top)
        x_hat = torch.cat([x_hat, z_bottom], dim=1)
        x_hat = self.final_dec(x_hat)
        x_hat = torch.tanh(x_hat)
        return x_hat

    def sample(self, t=None, sample_size=None, hw=None, mu=None, logvar=None, z=None):
        """
        not all the arguments are always necessary. the choices are:
        (sample size, hw) or (mu, logvar) or (z);
        and t <-> we use time conditioning (use_time=True in model init)

        :param t: the time step of the sample generated by the shader, tensor (B,)

        :param sample_size: Number of samples to be generated
        :param hw: int, side length of images to be generated

        :param mu: z means, see apply encoder. None for prior (init with zeros)
        :param logvar: z logstd, see apply encoder. None for prior (init with zeros)

        :param z: already sampled z (without adding time)

        :return: tensor of generated images, (B, 3, 256, 256)
        """
        if z is None:
            if mu is None:
                mu = (
                    torch.zeros((sample_size, self.latent_dim, hw//4, hw//4)).to(self.device),
                    torch.zeros((sample_size, self.latent_dim, hw//8, hw//8)).to(self.device)
                      )
            if logvar is None:
                logvar = (
                    torch.zeros((sample_size, self.latent_dim, hw//4, hw//4)).to(self.device),
                    torch.zeros((sample_size, self.latent_dim, hw//8, hw//8)).to(self.device)
                      )
            z = self.z_sample(mu, logvar)
        else:
            assert z[0].shape[1] == self.latent_dim, f"attempted to sample with z that already has time dimension." \
                                                     f"z_bottom={z[0].shape}, latent_dim={self.latent_dim}"

        if self.use_time:
            assert t is not None, "attempted to sample from CVAE without providing time variable"
            z = self.add_time_to_latent(z, t)

        x_hat = self.apply_decoder(z)
        return x_hat

    def add_time_to_latent(self, z, t):
        """
        concat a channel full of t to every pixel in both sampled z's
        """
        z_bottom, z_top = z
        hw_bottom = z_bottom.shape[-1]
        hw_top = z_top.shape[-1]
        bs = len(z_top)
        z_bottom = torch.cat([z_bottom, t.view(bs, 1, 1, 1).repeat((1, 1, hw_bottom, hw_bottom))], dim=1)
        z_top = torch.cat([z_top, t.view(bs, 1, 1, 1).repeat((1, 1, hw_top, hw_top))], dim=1)
        z = (z_bottom, z_top)
        return z

    def z_sample(self, mu, logvar):
        if type(mu) == tuple:
            assert type(logvar) == tuple and len(mu) == len(logvar)
            return tuple(self.z_sample(m, l) for m, l in zip(mu, logvar))

        std = torch.exp(logvar)
        # reparametrization trick
        eps = torch.randn(size=mu.size(), device=self.device)
        return eps*std + mu

    def forward(self, x, t):
        mu, logvar = self.apply_encoder(x)
        recon = self.sample(t, mu=mu, logvar=logvar)
        return recon, mu, logvar

    @torch.no_grad()
    def predict_video_from_image(self, image, fps=25, length_seconds=2, num_variations_to_generate=1):
        """
        image - tensor [0,1] image like (C,H,W) with dtype=torch.float32
        fps - frames per second
        length_seconds - video length in seconds
        num_variations_to_generate - sample z multiple times and gen video for each
        """
        self.eval()
        x = image.unsqueeze(0).to(device=self.device)  # turn to 1-size batch
        ts = torch.tensor(np.arange(length_seconds, step=1/fps), dtype=torch.float32, device=self.device)
        mu, logvar = self.apply_encoder(x)
        videos = []
        for i in range(num_variations_to_generate):
            z = self.z_sample(mu, logvar)

            # batch-fying z
            z_bottom, z_top = z
            z_bottom = torch.tile(z_bottom, (len(ts), 1, 1, 1))
            z_top = torch.tile(z_top, (len(ts), 1, 1, 1))
            z = (z_bottom, z_top)

            recon = self.sample(z=z, t=ts)
            video = tensor2np(recon, permute=False)
            # the array is (T, C, H, W) here. this is because wandb needs it that way for logging
            videos.append(video)
        return videos

    @torch.no_grad()
    def clear_grad_memory(self):
        # a trick to clear gpu RAM between training epochs
        for p in self.parameters():
            if p.grad is not None:
                del p.grad
        torch.cuda.empty_cache()
