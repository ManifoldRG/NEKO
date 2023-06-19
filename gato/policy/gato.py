import torch
import torch.nn as nn
from einops import rearrange

from gato.transformers import HFGPT
from gato.policy.embeddings import ImageEmbedding
from gato.policy.tokenizers import ContinuousTokenizer


class GatoPolicy(nn.Module):
    def __init__(
        self,
        embed_dim: int, 
        layers: int,
        heads: int,

        dropout: float,

        mu: int = 100,
        M: int = 256,

        patch_size: int = 16,
        resid_mid_channels: int = 128,
        num_groups: int = 32,
        position_vocab_size: int = 128,

        use_pos_encoding: bool = True,
        use_patch_pos_encoding: bool = True,

    ):
        # TODO, add option for disabling positional embeddings

        super().__init__()

        # this is a dummy value as this implementation does not yet handle language IO
        self.text_tokens = 32000 # SentencePiece vocab size
        self.continous_tokens = 1024
        self.discrete_tokens = 1024
        self.vocab_size = self.text_tokens + self.discrete_tokens + self.continous_tokens
        
        # order of text, continous, discrete
        self.token_starts = {
            'text': 0,
            'continous': self.text_tokens,
            'discrete': self.text_tokens + self.continous_tokens
        }

        self.embed_dim = embed_dim
        self.transformer = HFGPT(
            n_embd=embed_dim,
            use_geglu=True, # TODO, make this configurble
            n_layer=layers,
            n_head=heads,
            dropout=dropout,
            vocab_size=self.vocab_size,
        )
        # TODO, add option to init from pretrained LM

        self.separator_token = nn.Parameter(torch.zeros(embed_dim))

        # Tokenizers
        self.text_tokenizer = None # e.g. SentencePiece

        self.continuous_action_tokenizer = ContinuousTokenizer(
            use_mu_law=False, mu=mu, M=M, n_bins=self.continous_tokens, offset=self.token_starts['continous']
        ) # continuous actions expected to be in [-1, 1]
        
        self.continuous_obs_tokenizer = ContinuousTokenizer(
            use_mu_law=True, mu=mu, M=M, n_bins=self.continous_tokens, offset=self.token_starts['continous']
        ) # continuous actions expected to be in [-1, 1]


        # Token Embeddings
        self.embed_token = nn.Embedding(self.vocab_size, embed_dim)

        ## Image Embeddings
        self.image_embedding = ImageEmbedding(
            embed_dim=embed_dim,
            patch_size=patch_size,
            resid_mid_channels=resid_mid_channels,
            num_groups=num_groups,
            position_vocab_size=position_vocab_size,
            use_pos_encoding=self.use_patch_pos_encoding,
        )

        ## Inner-timestep Embeddings
        self.use_pos_encoding = use_pos_encoding
        self.pos_embed_observation = nn.Embedding()


    # predicts next token (for each input token)
    def forward(self, x):
        pass
    
    # generate the next n tokens
    def predict_n(self, x):
        pass

    # infer how many tokens needed to generate using environment, and restrict tokens generated to valid tokens for env
    def predict_control(self, x, env):
        pass