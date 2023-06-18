import torch
import torch.nn as nn
from einops import rearrange

import math


class ImageEmbedding(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            patch_size=16,
            resid_mid_channels=128,
            num_groups=32,
            position_vocab_size=128

    ):
        self.patch_size = patch_size
        self.patch_embedding = ResidualBlock_V2(mid_channels=resid_mid_channels, num_groups=num_groups)

        self.position_vocab_size position_vocab_size
        self.height_pos_embedding = nn.Embedding(position_vocab_size, embed_dim)
        self.width_pos_embedding = nn.Embedding(position_vocab_size, embed_dim)
    
    def forward(self, x, normalize=True):
        # reshape? (B x 1 x H x W) -> (B x 3 x H x W) if C = 1 TODO, probably do this before this function
        # assume all inputs have 3 channels, and dimensions are divisible by patch_size
        # all images in batch have same weight/width but network can handle inputs of different sizes through multiple forward passes

        image_height = x.shape[2]
        image_width = x.shape[3]

        assert image_height % self.patch_size == 0 and image_width % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        num_patches = (image_height // self.patch_size) * (image_width // self.patch_size)

        if normalize:
            # map from 0 to 255 to [-1,1], then scale by patch_size
            x = (x / 255.0 * 2) - 1
            x = x / math.sqrt(self.patch_size)
         
        # split into patches, rearrange
        x = rearrange(x, 'b c (n_h p) (n_w p) -> b n_h n_w c p p', p=self.patch_size)
         
        # number of patches along height,width
        n_height = x.shape[1]
        n_width = x.shape[2]

        training = False # TODO, infer this, or pass
        #training = True


        # compute intervals
        h_linspace = torch.linspace(0, 1, n_height + 1, device=x.device)
        w_linspace = torch.linspace(0, 1, n_width + 1, device=x.device)

        # start and end for each patch, along height and width
        h_intervals = torch.stack([h_linspace[:-1],h_linspace[1:]]).T # n_height x 2
        w_intervals = torch.stack([w_linspace[:-1],w_linspace[1:]]).T # n_width x 2

        # convert to integer (quantize)
        h_intervals = (h_intervals * self.position_vocab_size).to(dtype=torch.int32)
        w_intervals = (w_intervals * self.position_vocab_size).to(dtype=torch.int32)

        # sample from intervals or use mean
        if training:
            # verify this works
            # h_positions = torch.randint(h_intervals[:,0], h_intervals[:,1], size=(n_height,), device=x.device)
            # w_positions = torch.randint(w_intervals[:,0], w_intervals[:,1], size=(n_width,), device=x.device)
        else:
            # h_positions = torch.mean(h_intervals, dim=-1)
            # w_positions = torch.mean(w_intervals, dim=-1)
        
        # now get embeddings




class ResidualBlock_V2(nn.Module):

    def __init__(self, mid_channels: int = 128, num_groups: int = 32):

        super().__init__()
        in_channels = 3

        # Specific architecture not provided, potentially different
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.GeLU()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1) # Could do 1x1, 0 padding

        self.gn2 = nn.GroupNorm(num_groups, mid_channels)
        self.act2 = nn.GeLU()
        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # input: B x 3 x 16 x 16
        h = self.conv1(self.act1(self.g1(x)))
        h = self.conv2(self.act2(self.g2(h)))
        return x + h
