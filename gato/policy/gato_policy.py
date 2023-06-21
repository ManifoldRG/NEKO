import torch
import torch.nn as nn
from einops import rearrange

# import gato
from gato.transformers import HFGPT
from gato.policy.embeddings import ImageEmbedding
from gato.policy.input_tokenizers import ContinuousTokenizer


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
            n_positions=context,
        )
        # TODO, add option to init from pretrained LM

        # head
        self.predict_token = nn.Linear(embed_dim, self.vocab_size, bias=False)


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
        token_embeddings, tokens, token_target_masks, token_masks = self.tokenize_input_dicts(inputs)

        # pass to transformer
        output = self.transformer(x = token_embeddings, custom_mask = token_masks, batch_first=True)
        return output
        #import pdb; pdb.set_trace()


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
                continuous_actions: T x A or None
                discrete_actions: T x B or None 
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

        max_tokens = -1 # max number of timesteps across all batches
        for batch in inputs:
            text_tokens, text_embeddings, text_targets = None, None, None
            image_tokens, image_embeddings, image_targets = None, None, None
            continuous_tokens, continuous_embeddings, continuous_targets = None, None, None
            discrete_tokens, discrete_embeddings, discrete_targets = None, None, None
            continuous_action_tokens, continuous_action_embeddings, continuous_action_targets = None, None, None
            discrete_action_tokens, discrete_action_embeddings, discrete_action_targets = None, None, None

            n_timesteps = None

            # tokenize text
            if 'text' in batch and batch['text'] is not None:
                raise NotImplementedError
                text_tokens = self.text_tokenizer.tokenize(batch['text'])
                text_embeddings = self.embed_token(text_tokens)
                text_targets = torch.ones_like(text_tokens)
                n_timesteps = text_tokens.shape[0]
                # batch_ids.append(text_tokens)
                # batch_embeddings.append(text_embeddings)
                # batch_targets.append(torch.ones_like(text_tokens))

            if 'images' in batch and batch['images'] is not None:
                image_embeddings = self.image_embedding(batch['images']) # n_timesteps x n_patches x embed_dim
                n_images = image_embeddings.shape[0]
                n_patches = image_embeddings.shape[1]
                image_tokens = torch.ones(n_images, n_patches) * -1
                image_targets = torch.zeros(n_images, n_patches)
                if n_timesteps is None:
                    n_timesteps = n_images
                else:
                    assert n_timesteps == n_images, "number of timesteps must be the same for all modalities"
            
            if 'continuous' in batch and batch['continuous'] is not None:
                continuous_tokens = self.continuous_obs_tokenizer.tokenize(batch['continuous'])
                continuous_embeddings = self.embed_token(continuous_tokens)
                continuous_targets = torch.zeros_like(continuous_tokens)

                if n_timesteps is None:
                    n_timesteps = continuous_tokens.shape[0]
                else:
                    assert n_timesteps == continuous_tokens.shape[0], "number of timesteps must be the same for all modalities"
            
            if 'discrete' in batch and batch['discrete'] is not None:
                discrete_tokens = batch['discrete']
                discrete_embeddings = self.embed_token(discrete_tokens)
                discrete_targets = torch.zeros_like(discrete_tokens)

                if n_timesteps is None:
                    n_timesteps = discrete_tokens.shape[0]
                else:
                    assert n_timesteps == discrete_tokens.shape[0], "number of timesteps must be the same for all modalities"
            
            if 'continuous_actions' in batch and batch['continuous_actions'] is not None:
                continuous_action_tokens = self.continuous_action_tokenizer.tokenize(batch['continuous_actions'])
                continuous_action_embeddings = self.embed_token(continuous_action_tokens)
                continuous_action_targets = torch.ones_like(continuous_action_tokens)
                
                if n_timesteps is None:
                    n_timesteps = continuous_action_tokens.shape[0]
                else:
                    assert n_timesteps == continuous_action_tokens.shape[0], "number of timesteps must be the same for all modalities"

            if 'discrete_actions' in batch and batch['discrete_actions'] is not None:
                discrete_action_tokens = batch['discrete_actions']
                discrete_action_embeddings = self.embed_token(discrete_action_tokens)
                discrete_action_targets = torch.ones_like(discrete_action_tokens)

                if n_timesteps is None:
                    n_timesteps = discrete_action_tokens.shape[0]
                else:
                    assert n_timesteps == discrete_action_tokens.shape[0], "number of timesteps must be the same for all modalities"

            

            separator_embeddings = torch.ones(n_timesteps, 1, self.embed_dim) * self.separator_token
            separator_tokens = torch.ones(n_timesteps, 1) * -1
            separator_targets = torch.zeros(n_timesteps, 1)
            
            # interleave observation, action tokens,add separator

            # interleave tokens
            batch_tokens = torch.cat(
                [
                    tokens for tokens in
                    [text_tokens, image_tokens, continuous_tokens, discrete_tokens, separator_tokens, continuous_action_tokens, discrete_action_tokens] 
                    if tokens is not None
                ], 
                dim=1
            )

            # interleave targets
            batch_target_masks = torch.cat(
                [
                    targets for targets in 
                    [text_targets, image_targets, continuous_targets, discrete_targets, separator_targets, continuous_action_targets, discrete_action_targets] 
                    if targets is not None
                ], 
                dim=1
            )

            # interleave embeddings, n_timesteps x n_tokens x embed_dim
            batch_embeddings = torch.cat(
                [
                    embeddings for embeddings in
                    [text_embeddings, image_embeddings, continuous_embeddings, discrete_embeddings]
                    if embeddings is not None
                ], 
                dim=1
            ) # concat observations

            n_observation_tokens = batch_embeddings.shape[1] # number of tokens per timestep
            if self.use_pos_encoding:
                inner_timestep_embeddings = self.pos_embed_observation(torch.arange(n_observation_tokens)).unsqueeze(0) # 1 x n_tokens x embed_dim
                # repeat for each timestep
                inner_timestep_embeddings = inner_timestep_embeddings.repeat(n_timesteps, 1, 1)
                batch_embeddings = batch_embeddings + inner_timestep_embeddings

            action_embeddings = torch.cat([action_embedding for action_embedding in [continuous_action_embeddings, discrete_action_embeddings] if action_embedding is not None], dim=1) # concat action
            batch_embeddings = torch.cat([batch_embeddings, separator_embeddings, action_embeddings], dim=1) # concat action and separator
            
            tokens_per_timestep = batch_embeddings.shape[1] # number of tokens per timestep
            total_tokens = n_timesteps * tokens_per_timestep
            
            # reshape to 1 x (n_timesteps * n_tokens) x embed_dim
            batch_embeddings = batch_embeddings.reshape(1, total_tokens, self.embed_dim)
            batch_tokens = batch_tokens.reshape(1, total_tokens)
            batch_target_masks = batch_target_masks.reshape(1, total_tokens)
            total_tokens = batch_embeddings.shape[1]
            max_tokens = max(max_tokens, total_tokens)

            token_embeddings.append(batch_embeddings)
            tokens.append(batch_tokens)
            token_target_masks.append(batch_target_masks)

        token_masks = []
        # (left pad) to max tokens
        for i in range(n_batches):
            # store which tokens are padding and which are real
            token_masks.append(torch.cat([torch.zeros(1, max_tokens -  token_embeddings[i].shape[1]), torch.ones(1,  token_embeddings[i].shape[1])], dim=1))
            
            batch_embeddings = torch.cat([torch.zeros(1, max_tokens - token_embeddings[i].shape[1], self.embed_dim), token_embeddings[i]], dim=1)
            batch_tokens = torch.cat([torch.ones(1, max_tokens - tokens[i].shape[1]) * -1, tokens[i]], dim=1)
            batch_target_masks = torch.cat([torch.zeros(1, max_tokens - token_target_masks[i].shape[1]), token_target_masks[i]], dim=1)

        # concat
        token_embeddings = torch.cat(token_embeddings, dim=0)
        tokens = torch.cat(tokens, dim=0)
        token_target_masks = torch.cat(token_target_masks, dim=0)
        token_masks = torch.cat(token_masks, dim=0)
        return token_embeddings, tokens, token_target_masks, token_masks
    
    # generate the next n tokens
    def predict_n(self, x):
        pass

    # infer how many tokens needed to generate using environment, and restrict tokens generated to valid tokens for env
    def predict_control(self, x, env):
        pass

if __name__ == '__main__':
    model = GatoPolicy(
        embed_dim=128,
        layers=2,
        heads=4,

        dropout=0.1,

        patch_size=16,
        resid_mid_channels=128,
        num_groups=32,
    )
    n_timesteps = 24
    inputs = [{
        #'images': torch.randn(10, 3, 224, 224),
        'images': torch.randn(n_timesteps, 3, 80, 64),
        'discrete_actions': torch.randint(0, 4, (n_timesteps, 1)),
    }]

    output = model(inputs)
    import pdb; pdb.set_trace()