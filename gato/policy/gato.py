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
        continuous_tokens: int = 1024,
        discrete_tokens: int = 1024,

        context=1024,

        use_pos_encoding: bool = True,
        use_patch_pos_encoding: bool = True,
        

    ):
        # TODO, add option for disabling positional embeddings

        super().__init__()

        self.context = context
        # this is a dummy value as this implementation does not yet handle language IO
        self.text_tokens = 32000 # SentencePiece vocab size
        self.continous_tokens = continuous_tokens
        self.discrete_tokens = discrete_tokens
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
        self.use_patch_pos_encoding = use_patch_pos_encoding
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
        self.pos_embed_observation = nn.Embedding(context, embed_dim)


    # predicts next token (for each input token)
    def forward(self, inputs):
        batch_size = len(inputs)


        pass

    def tokenize_input_dicts(self, inputs):
        """"
        inputs: list of dicts for each batch
        [
            {
                # observations
                text: T x L  or None
                images: T x 3 x H x W or None
                continuous: T x C or None # continuous vector observations
                discrete: T x D or None   # discrete observations

                # actions
                actions: T x A or None 
            },
            ...
            {
            }
        ]

        returns: the tokens_id, tokens_embedding for each batch respectively

        token_embedding:
        [
            tensor 1 x ? x embed_dim
            tensor 1 x ? x embed_dim
            tensor 1 x ? x embed_dim
            ...
        ] where ? represents the variable number of tokens for each batch

        token_id:
        [
            tensor 1 x ? x 1
            tensor 1 x ? x 1
            tensor 1 x ? x 1
            ...
        ] where each element is the token id for each token embedding, which is set to -1 for image tokens

        token_target:
        [
            tensor 1 x ? x 1
            tensor 1 x ? x 1
            tensor 1 x ? x 1
        ] # binary mask for each token, 1 if token is a predicted target token, 0 otherwise
        # text observation and continuous actions are predicted, while images and observation tensors are not
        """
        n_batches = len(inputs)

        token_embeddings = []
        tokens = []
        token_target_masks = []

        for batch in inputs:
            text_tokens, text_embeddings, text_targets = None, None, None
            image_tokens, image_embeddings, image_targets = None, None, None
            continuous_tokens, continuous_embeddings, continuous_targets = None, None, None
            discrete_tokens, discrete_embeddings, discrete_targets = None, None, None
            action_tokens, action_embeddings, action_targets = None, None, None

            n_timesteps = None

            # tokenize text
            if batch['text'] is not None:
                raise NotImplementedError
                text_tokens = self.text_tokenizer.tokenize(batch['text'])
                text_embeddings = self.embed_token(text_tokens)
                text_targets = torch.ones_like(text_tokens)
                n_timesteps = text_tokens.shape[0]
                # batch_ids.append(text_tokens)
                # batch_embeddings.append(text_embeddings)
                # batch_targets.append(torch.ones_like(text_tokens))

            if batch['images'] is not None:
                image_embeddings = self.image_embedding(batch['images']).unsqueeze(0)
                n_images = image_embeddings.shape[0]
                n_patches = image_embeddings.shape[1]
                image_tokens = torch.ones(n_images, n_patches, 1) * -1
                image_targets = torch.zeros(n_images, n_patches, 1)

                if n_timesteps is None:
                    n_timesteps = n_images
                else:
                    assert n_timesteps == n_images, "number of timesteps must be the same for all modalities"
            
            if batch['continuous'] is not None:
                continuous_tokens = self.continuous_obs_tokenizer.tokenize(batch['continuous'])
                continuous_embeddings = self.embed_token(continuous_tokens)
                continuous_targets = torch.zeros_like(continuous_tokens)

                if n_timesteps is None:
                    n_timesteps = continuous_tokens.shape[0]
                else:
                    assert n_timesteps == continuous_tokens.shape[0], "number of timesteps must be the same for all modalities"
            
            if batch['discrete'] is not None:
                discrete_tokens = batch['discrete']
                discrete_embeddings = self.embed_token(discrete_tokens)
                discrete_targets = torch.zeros_like(discrete_tokens)

                if n_timesteps is None:
                    n_timesteps = discrete_tokens.shape[0]
                else:
                    assert n_timesteps == discrete_tokens.shape[0], "number of timesteps must be the same for all modalities"
            
            if batch['actions'] is not None:
                action_tokens = self.continuous_action_tokenizer.tokenize(batch['actions'])
                action_embeddings = self.embed_token(action_tokens)
                action_targets = torch.ones_like(action_tokens)

                if n_timesteps is None:
                    n_timesteps = action_tokens.shape[0]
                else:
                    assert n_timesteps == action_tokens.shape[0], "number of timesteps must be the same for all modalities"

            separator_embeddings = torch.ones(n_timesteps, 1, self.embed_dim) * self.separator_token
            separator_tokens = torch.ones(n_timesteps, 1) * -1
            separator_targets = torch.zeros(n_timesteps, 1)
            
            # interleave observation, action tokens,add separator

            # interleave tokens
            batch_tokens = torch.cat([text_tokens, image_tokens, continuous_tokens, discrete_tokens, separator_tokens, action_tokens], dim=1)

            # interleave targets
            batch_target_masks = torch.cat([text_targets, image_targets, continuous_targets, discrete_targets, separator_targets, action_targets], dim=1)

            # interleave embeddings, n_timesteps x n_tokens x embed_dim
            batch_embeddings = torch.cat([text_embeddings, image_embeddings, continuous_embeddings, discrete_embeddings], dim=1) # concat observations
            n_observation_tokens = batch_embeddings.shape[1] # number of tokens per timestep
            if self.use_pos_encoding:
                inner_timestep_embeddings = self.pos_embed_observation(torch.arange(n_observation_tokens)).unsqueeze(0) # 1 x n_tokens x embed_dim
                # repeat for each timestep
                inner_timestep_embeddings = inner_timestep_embeddings.repeat(n_timesteps, 1, 1)
                batch_embeddings = batch_embeddings + inner_timestep_embeddings

            batch_embeddings = torch.cat([batch_embeddings, separator_embeddings, action_embeddings], dim=1) # concat action and separator
            
            n_tokens = batch_embeddings.shape[1] # number of tokens per timestep

            token_embeddings.append(batch_embeddings)
            tokens.append(batch_tokens)
            token_target_masks.append(batch_target_masks)

        # TODO, potentially pad / concat lists, return tensors 
        return token_embeddings, tokens, token_target_masks
    
    # generate the next n tokens
    def predict_n(self, x):
        pass

    # infer how many tokens needed to generate using environment, and restrict tokens generated to valid tokens for env
    def predict_control(self, x, env):
        pass