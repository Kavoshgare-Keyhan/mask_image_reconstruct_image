# vqvae.py (edited minimally — fixes forward/encode outputs to 4 items)
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# (Keep your distributed helper imports if needed)
# import distributed as dist_fn

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)  # shape (dim, n_embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
        self.initialized = True

    def forward(self, input):
        if not self.initialized:
            self.initialize_codebook(input)

        flatten = input.reshape(-1, self.dim)  # (N, dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)  # (N,)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])  # shape (... spatial)
        quantize = self.embed_code(embed_ind)

        num_used_codebooks = -1
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # Distributed averaging if you have distributed utilities:
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n)
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            unique_indices = torch.unique(torch.flatten(embed_ind), sorted=False)
            num_used_codebooks = unique_indices.numel()

            # Optionally reinit unused embeddings if needed (kept minimal)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind, num_used_codebooks

    def embed_code(self, embed_id):
        # embed_id of shape (...); self.embed is (dim, n_embed); transpose -> (n_embed, dim)
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def initialize_codebook(self, input):
        flatten = input.reshape(-1, self.dim)
        self.embed.data.copy_(flatten[: self.n_embed].t())
        self.embed_avg.data.copy_(self.embed.data)
        self.initialized = True


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = []
        if stride == 4:
            blocks.extend([
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ])
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class FlatVQVAE(nn.Module):
    def __init__(self, in_channel=3, channel=144, n_res_block=2, n_res_channel=72,
                 embed_dim=144, n_embed=456, decay=0.99):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_b = Quantize(embed_dim, n_embed, decay=decay)
        self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.vocab_size = n_embed

    def encode(self, x):
        enc_b = self.enc_b(x)  # (B, embed_dim, H, W) maybe channel alignment required
        # your original permute: bring channel dim to last for quantize
        quant_in = enc_b.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        quant_b, diff_b, id_b, num_used = self.quantize_b(quant_in)  # quant_b same shape as quant_in
        # get quant back to (B,C,H,W)
        quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
        return quant_b, diff_b, id_b, num_used

    def decode(self, quant_b):
        return self.dec(quant_b)

    def decode_code(self, code_b):
        # code_b: (B,H,W) integer ids
        quant_b = self.quantize_b.embed_code(code_b)  # returns (B,H,W,dim)
        quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
        return self.decode(quant_b)

    def forward(self, x):
        quant_b, diff_b, id_b, num_used = self.encode(x)
        dec = self.decode(quant_b)
        # Return exactly 4 outputs: (reconstruction, latent loss, indices, num_used)
        return dec, diff_b, id_b, num_used
