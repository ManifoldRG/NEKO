# Test replacing ImageEmbedding with the ViT from https://github.com/lucidrains/vit-pytorch
import torch
import torch.nn as nn
import math
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

class ImageEmbedding(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            patch_size=16
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        vit = ViT(
                image_size = 256,
                patch_size = patch_size,
                num_classes = 1000,
                dim = embed_dim,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
        )
        self.v = Extractor(vit)

    def forward(self, x, normalize=True):     
        if normalize:
            # map from 0 to 255 to [-1,1], then scale by patch_size
            x = (x / 255.0 * 2) - 1
            x = x / math.sqrt(self.patch_size)
        embeddings = self.v(x, return_embeddings_only = True)
        return embeddings[:, 1:, :] # the first token os for CLS, need to remove it, so we use 1: