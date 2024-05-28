import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

from einops import rearrange
import math

class Pretrained_ImageEmbedding(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            patch_size=16,  
            # the batch_size and embed_dim passed in here are just placeholders, the embedding model used below will use the deame default value 768 and 16
            model_name = 'vit_base_patch16_224'  # You can choose different variants, patch size is 16*16, image size is 224*224

    ):
        super().__init__()
        # Load a pre-trained Vision Transformer model
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()  # Set the model to evaluation mode

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to the required input size
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),  # Normalization mean (ImageNet values)
                std=(0.229, 0.224, 0.225)    # Normalization std (ImageNet values)
            ),
        ])

        self.embedding = None
        # Define a hook function to capture the embedding
        def hook(module, input, output):
            global embedding
            embedding = output

        # Register the hook to the patch embedding layer. 
        self.hook_handle = self.model.patch_embed.register_forward_hook(hook)
        
        # After all data processed, remember to call image_embedding.cleanup() which is defined below 
        # to unregister the hook, assuming the instantiated object from this class is named "image_embedding" 
        # But this might not be necessary because image_embedding might need to be used even after all data processed?
        def cleanup(self):
            #unregister the hook
            self.hook_handle.remove()

    def forward(self, img): # the img here is the orignal image returned from Image.open(), it is not the image data tensor
        img = self.transform(img).unsqueeze(0)  # Add a batch dimension
        # Perform a forward pass to trigger the hook
        with torch.no_grad():
            _ = self.model(img)

        # Return the captured embedding
        return embedding

class ImageEmbedding(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            patch_size=16,
            resid_mid_channels=128,
            num_groups=32,
            position_vocab_size=128,
            use_pos_encoding=True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = ResidualBlock_V2(mid_channels=resid_mid_channels, num_groups=num_groups)
        self.post_embedding_projection = nn.Linear(patch_size * patch_size * 3, embed_dim)

        self.use_pos_encoding = use_pos_encoding
        self.patch_pos_encoding = PatchPosEncoding(position_vocab_size=position_vocab_size, embed_dim=embed_dim)
    
    def forward(self, x, normalize=True):
        # reshape? (B x 1 x H x W) -> (B x 3 x H x W) if C = 1 TODO, probably do this before this function
        # all images in batch must have same weight/width but network can handle inputs of different sizes through multiple forward passes
        image_height = x.shape[2]
        image_width = x.shape[3]

        assert image_height % self.patch_size == 0 and image_width % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        n_height = image_height // self.patch_size
        n_width = image_width // self.patch_size

        if normalize:
            # map from 0 to 255 to [-1,1], then scale by patch_size
            x = (x / 255.0 * 2) - 1
            x = x / math.sqrt(self.patch_size)
         
        # split into patches, rearrange
        x = rearrange(x, 'b c (n_h p_1) (n_w p_2) -> (b n_h n_w) c p_1 p_2', p_1=self.patch_size, p_2=self.patch_size)

        # embed patches
        x = self.patch_embedding(x)

        # rearrange again:
        x = rearrange(x, '(b n_h n_w) c p_1 p_2 -> b n_h n_w (c p_1 p_2)', p_1=self.patch_size, p_2=self.patch_size, n_h=n_height, n_w=n_width)

        # post linear projection
        x = self.post_embedding_projection(x) # b n_h n_w embed_dim
         
        # now add positional encoding
        if self.use_pos_encoding:
            x = x + self.patch_pos_encoding(x)
            
        x = rearrange(x, 'b n_h n_w embed_dim -> b (n_h n_w) embed_dim')

        return x

class PatchPosEncoding(nn.Module):
    def __init__(self, position_vocab_size=128, embed_dim=768):
        super().__init__()

        self.position_vocab_size = position_vocab_size
        self.embed_dim = embed_dim
        self.height_pos_embedding = nn.Embedding(position_vocab_size, embed_dim)
        self.width_pos_embedding = nn.Embedding(position_vocab_size, embed_dim)
        
    def forward(self, x):
        # x: B x n_height x n_width x embed_dim

        # number of patches along height,width
        n_height = x.shape[1]
        n_width = x.shape[2]

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
        if self.training:
            # sample from interval
            h_positions = torch.tensor([torch.randint(low=interval[0], high=interval[1], size=()) for interval in h_intervals], device=x.device)
            w_positions = torch.tensor([torch.randint(low=interval[0], high=interval[1], size=()) for interval in w_intervals], device=x.device)
        else:
            h_intervals[:, 1] = h_intervals[:, 1] - 1
            w_intervals[:, 1] = w_intervals[:, 1] - 1
            h_positions = h_intervals.mean(dim=-1,dtype=torch.float32).round().to(dtype=torch.int32)
            w_positions = w_intervals.mean(dim=-1,dtype=torch.float32).round().to(dtype=torch.int32)
        # now get embeddings
        h_position_embed = self.height_pos_embedding(h_positions) # n_height x embed_dim
        w_position_embed = self.width_pos_embedding(w_positions) # n_width x embed_dim

        # combine height, width embeddings
        h_position_embed = h_position_embed.unsqueeze(1).repeat(1, n_width, 1) # n_height x n_width x embed_dim
        w_position_embed = w_position_embed.unsqueeze(0).repeat(n_height, 1, 1) # n_height x n_width x embed_dim

        position_embed = h_position_embed + w_position_embed
        return position_embed
class ResidualBlock_V2(nn.Module):

    def __init__(self, mid_channels: int = 128, num_groups: int = 32):
        super().__init__()
        in_channels = 3

        # Specific architecture not provided, potentially different
        #self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.gn1 = nn.Identity()
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1) # Could do 1x1, 0 padding

        self.gn2 = nn.GroupNorm(num_groups, mid_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # input: B x 3 x 16 x 16
        h = self.conv1(self.act1(self.gn1(x)))
        h = self.conv2(self.act2(self.gn2(h)))
        return x + h
        