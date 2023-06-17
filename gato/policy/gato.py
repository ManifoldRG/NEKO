import torch
import torch.nn as nn
from einops import rearrange

from gato.gpt import HFGPT


class GatoPolicy(nn.Module):
    def __init__(
        self,
        embed_dim: int, 
        layers: int,
        heads: int,
        dropout: float
    ):
        
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


        # Embeddings
        self.embed_token = nn.Embedding(self.vocab_size, embed_dim)

        ## Image Embeddings


        ## Inner-timestep Embeddings
        self.observation_pos_embd




    # predicts next token (for each input token)
    def forward(self, x):
        pass
    
    # generate the next n tokens
    def predict_n(self, x):
        pass