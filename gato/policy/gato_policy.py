import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

import gymnasium as gym
import transformers

# import gato
from gato.transformers import GPT2Model
from gato.policy.embeddings import ImageEmbedding
from gato.policy.input_tokenizers import ContinuousTokenizer
from gato.tasks.control_task import ControlTask

class GatoPolicy(nn.Module):
    def __init__(
        self,
        device: str,
        embed_dim: int, 
        layers: int,
        heads: int,
        dropout: float,

        activation_fn='gelu',

        mu: int = 100,
        M: int = 256,

        patch_size: int = 16,
        resid_mid_channels: int = 132,
        num_groups: int = 32,
        position_vocab_size: int = 128,
        continuous_tokens: int = 1024,
        discrete_tokens: int = 1024,

        context_len=1024,

        use_pos_encoding: bool = True,
        use_patch_pos_encoding: bool = True,

        pretrained_lm: str = None, # Optional, name of pretrained language model to use
        flash: bool = False, # TODO, verify correctness
        pad_seq: bool = False
    ):
        super().__init__()

        self.device = device

        self.context_len = context_len
        self.pad_seq = pad_seq
        # this is a dummy value as this implementation does not yet handle language IO
        #self.text_tokens = 32000 # SentencePiece vocab size
        self.text_tokens = 1
        self.continuous_tokens = continuous_tokens
        self.discrete_tokens = discrete_tokens
        self.vocab_size = self.text_tokens + self.discrete_tokens + self.continuous_tokens
        
        # order of text, continuous, discrete
        self.token_starts = {
            'text': 0,
            'continuous': self.text_tokens,
            'discrete': self.text_tokens + self.continuous_tokens
        }

        self.token_ends = {
            'text': self.text_tokens - 1,
            'continuous': self.text_tokens + self.continuous_tokens - 1,
            'discrete': self.text_tokens + self.continuous_tokens + self.discrete_tokens - 1
        }


        # self.transformer = HFGPT(
        #     n_embd=embed_dim,
        #     n_layer=layers,
        #     n_head=heads,
        #     dropout=dropout,
        #     vocab_size=self.vocab_size,
        #     n_positions=context_len,
        #     activation_fn=activation_fn,
        # )
        if pretrained_lm is not None:
            config = transformers.GPT2Config.from_pretrained(pretrained_lm)
            config.attn_pdrop = dropout # 0.1
            config.resid_pdrop = dropout
            config.flash = flash
            config.gate = False
            self.transformer = GPT2Model.from_pretrained(
                pretrained_lm,
                config=config,
            )
            embed_dim = config.n_embd
        else:
            gate = False
            if activation_fn == 'geglu':
                gate = True
                activation_fn = 'gelu'
            config = transformers.GPT2Config(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=embed_dim,
                n_head=heads,
                n_layer=layers,
                resid_pdrop=dropout,
                attn_pdrop=dropout,
                n_positions=context_len,
                n_inner=embed_dim * 4,
                activation_function=activation_fn,
            )
            config.flash = flash
            config.n_ctx = context_len
            config.gate = gate
            self.transformer = self.transformer = GPT2Model(config)

        self.embed_dim = embed_dim

        # head
        self.predict_token = nn.Linear(embed_dim, self.vocab_size, bias=False)


        self.separator_token = nn.Parameter(torch.zeros(embed_dim))

        # Tokenizers
        self.text_tokenizer = None # e.g. SentencePiece

        self.continuous_action_tokenizer = ContinuousTokenizer(
            use_mu_law=False, mu=mu, M=M, n_bins=self.continuous_tokens, offset=self.token_starts['continuous']
        ) # continuous actions expected to be in [-1, 1]
        
        self.continuous_obs_tokenizer = ContinuousTokenizer(
            use_mu_law=True, mu=mu, M=M, n_bins=self.continuous_tokens, offset=self.token_starts['continuous']
        )


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
        self.pos_embed_observation = nn.Embedding(context_len, embed_dim)



    # predicts next token (for each input token)
    def forward(self, inputs: list = None, compute_loss=False, use_cache=False, past_key_values=None, **kwargs):
        # tokenize inputs
        if inputs is not None:
            token_embeddings, tokens, token_target_masks, token_masks = self.tokenize_input_dicts(inputs)
        else:
            assert 'token_embeddings' in kwargs and 'tokens' in kwargs and 'token_target_masks' in kwargs and 'token_masks' in kwargs, 'if inputs is None, must provide embeddings, tokens, and masks'
            token_embeddings = kwargs['token_embeddings']
            tokens = kwargs['tokens']
            token_target_masks = kwargs['token_target_masks']
            token_masks = kwargs['token_masks']

        # pass to transformer
        #final_representations = self.transformer(x = token_embeddings, custom_mask = token_masks, batch_first=True)
        if past_key_values is not None:
            # extend attention mask to account for number of past key values
            token_masks = torch.cat([token_masks, torch.ones(token_masks.shape[0], past_key_values[0].shape[-2]).to(token_masks.device)], dim=1)
        
        transformer_output = self.transformer(
            inputs_embeds=token_embeddings, 
            attention_mask=token_masks, 
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        final_representations = transformer_output['last_hidden_state']
        if use_cache:
            past_key_values = list(transformer_output['past_key_values'])
        else:
            past_key_values = None

        # predict logits
        logits = self.predict_token(final_representations)

        if compute_loss:
            # obtain target tokens, and pad
            loss_logits = logits[:, :-1, :]
            token_masks = token_masks[:, :-1] # whether originating token is valid

            token_target_masks = token_target_masks[:, 1:] # whether target token is valid
            loss_masks = token_masks * token_target_masks
            target_tokens = tokens[:, 1:]

            loss_masks = loss_masks.reshape(-1)
            loss_logits = loss_logits.reshape(-1, self.vocab_size)[loss_masks > 0]
            target_tokens = target_tokens.reshape(-1)[loss_masks > 0]
            loss = torch.nn.functional.cross_entropy(loss_logits, target_tokens)
            if 'pdb' in kwargs and kwargs['pdb']:
                import pdb; pdb.set_trace()
        else:
            loss = None
        
        return logits, loss, past_key_values


    def tokenize_input_dicts(self, inputs: list):
        """"
        inputs: list of dicts for each batch
        [
            {
                # observations
                text: T x L  or None
                images: T x 3 x H x W or None
                continuous_obs: T x C or None # continuous vector observations
                discrete_obs: T x D or None   # discrete observations

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
                #image_tokens = torch.ones(n_images, n_patches) * -1
                image_tokens = torch.zeros(n_images, n_patches, dtype=torch.long, device=self.device)
                image_targets = torch.zeros(n_images, n_patches, device=self.device)
                if n_timesteps is None:
                    n_timesteps = n_images
                else:
                    assert n_timesteps == n_images, "number of timesteps must be the same for all modalities"
            
            if 'continuous_obs' in batch and batch['continuous_obs'] is not None:
                continuous_tokens = self.continuous_obs_tokenizer.encode(batch['continuous_obs'])
                continuous_embeddings = self.embed_token(continuous_tokens)
                continuous_targets = torch.zeros_like(continuous_tokens, device=self.device)

                if n_timesteps is None:
                    n_timesteps = continuous_tokens.shape[0]
                else:
                    assert n_timesteps == continuous_tokens.shape[0], "number of timesteps must be the same for all modalities"
            
            if 'discrete_obs' in batch and batch['discrete_obs'] is not None:
                discrete_tokens = batch['discrete_obs']
                discrete_tokens = discrete_tokens + self.token_starts['discrete'] # add offset
                discrete_embeddings = self.embed_token(discrete_tokens)
                discrete_targets = torch.zeros_like(discrete_tokens, device=self.device)

                if n_timesteps is None:
                    n_timesteps = discrete_tokens.shape[0]
                else:
                    assert n_timesteps == discrete_tokens.shape[0], "number of timesteps must be the same for all modalities"
            
            if 'continuous_actions' in batch and batch['continuous_actions'] is not None:
                continuous_action_tokens = self.continuous_action_tokenizer.encode(batch['continuous_actions'])
                continuous_action_embeddings = self.embed_token(continuous_action_tokens)
                continuous_action_targets = torch.ones_like(continuous_action_tokens, device=self.device)
                
                if n_timesteps is None:
                    n_timesteps = continuous_action_tokens.shape[0]
                else:
                    assert n_timesteps == continuous_action_tokens.shape[0], "number of timesteps must be the same for all modalities"

            if 'discrete_actions' in batch and batch['discrete_actions'] is not None:
                discrete_action_tokens = batch['discrete_actions']
                discrete_action_tokens = discrete_action_tokens + self.token_starts['discrete'] # add offset

                # embed
                discrete_action_embeddings = self.embed_token(discrete_action_tokens)
                discrete_action_targets = torch.ones_like(discrete_action_tokens)

                if n_timesteps is None:
                    n_timesteps = discrete_action_tokens.shape[0]
                else:
                    assert n_timesteps == discrete_action_tokens.shape[0], "number of timesteps must be the same for all modalities"

            

            separator_embeddings = torch.ones(n_timesteps, 1, self.embed_dim, device=self.device) * self.separator_token
            separator_tokens = torch.zeros(n_timesteps, 1, dtype=torch.long, device=self.device)
            separator_targets = torch.zeros(n_timesteps, 1, dtype=torch.long, device=self.device)
            
            # interleave observation, action tokens,add separator

            # interleave tokens
            batch_tokens = torch.cat(
                [
                    tokens for tokens in
                    [text_tokens, image_tokens, continuous_tokens, discrete_tokens, separator_tokens, continuous_action_tokens, discrete_action_tokens] 
                    if tokens is not None
                ], 
                dim=1,
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
                inner_timestep_embeddings = self.pos_embed_observation(torch.arange(n_observation_tokens, device=self.device)).unsqueeze(0) # 1 x n_tokens x embed_dim
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
            token_masks.append(torch.cat([torch.zeros(1, max_tokens -  token_embeddings[i].shape[1], device=self.device), torch.ones(1,  token_embeddings[i].shape[1], device=self.device)], dim=1))
            
            token_embeddings[i] = torch.cat([torch.zeros(1, max_tokens - token_embeddings[i].shape[1], self.embed_dim, device=self.device), token_embeddings[i]], dim=1)
            tokens[i] = torch.cat([torch.zeros(1, max_tokens - tokens[i].shape[1], dtype=torch.long, device=self.device), tokens[i]], dim=1)
            token_target_masks[i] = torch.cat([torch.zeros(1, max_tokens - token_target_masks[i].shape[1], device=self.device), token_target_masks[i]], dim=1)

        # concat
        token_embeddings = torch.cat(token_embeddings, dim=0)
        tokens = torch.cat(tokens, dim=0)
        token_target_masks = torch.cat(token_target_masks, dim=0)
        token_masks = torch.cat(token_masks, dim=0)

        if self.pad_seq:
            # get seq length
            seq_len = token_embeddings.shape[1]
            pad_len = self.context_len - seq_len
            if pad_len > 0:
                token_embeddings = torch.cat([token_embeddings, torch.zeros(n_batches, pad_len, self.embed_dim, device=self.device)], dim=1)
                tokens = torch.cat([tokens, torch.zeros(n_batches, pad_len, dtype=torch.long, device=self.device)], dim=1)
                token_target_masks = torch.cat([token_target_masks, torch.zeros(n_batches, pad_len, device=self.device)], dim=1)
                token_masks = torch.cat([token_masks, torch.zeros(n_batches, pad_len, device=self.device)], dim=1)
        return token_embeddings, tokens, token_target_masks, token_masks
    

    # infer how many tokens needed to generate using environment, and restrict tokens generated to valid tokens for env
    def predict_control(self, input: dict, task: ControlTask, deterministic: bool = True, past_key_values=None, use_cache=False, **kwargs):
        # expects that inputs['continuous_actions'] or inputs['discrete_actions'] are padded by 1 timestep
        
        action_type = task.action_type # continuous or discrete
        action_tokens = task.action_tokens

        if action_type == gym.spaces.Discrete:
            action_str = 'discrete'
            assert action_tokens == 1, "only support 1 discrete action token"
        elif action_type == gym.spaces.Box:
            action_str = 'continuous'
        
        start_token = self.token_starts[action_str]
        end_token = self.token_ends[action_str]

        # further restrict end_token if discrete action
        if action_str == 'discrete':
            assert task.env.action_space.n <= self.discrete_tokens, "discrete action space too large for model"
            end_token = start_token + task.env.action_space.n - 1
        token_embeddings, _, _, token_masks = self.tokenize_input_dicts([input])

        # remove last action_tokens tokens, which are padding

        token_embeddings = token_embeddings[:, :-action_tokens, :]
        token_masks = token_masks[:, :-action_tokens]
        
        predicted_tokens = []
        
        #import pdb; pdb.set_trace()
        # truncate past_key_values to context len
        #import pdb; pdb.set_trace()
        if past_key_values is not None: 
            # and trim off fisrst obs + padding tokens
            #trim_tokens = task.tokens_per_timestep - action_tokens
            trim_tokens = task.tokens_per_timestep - 1 # trim everything except last action token from first timestep
            trim_tokens = task.tokens_per_timestep - action_tokens
            
            token_embeddings = token_embeddings[:, trim_tokens:, :]
            token_masks = token_masks[:, trim_tokens:]
            
            # trim prev action tokens
            for i in range(len(past_key_values)):
                past_key_values[i] = past_key_values[i][:, :, :, :-(action_tokens - 1), :]

            print('input tokens', token_embeddings.shape)
            #diff = (past_key_values[0].shape[-2] + token_embeddings.shape[1] + action_tokens - 1) - self.context_len
            diff = (past_key_values[0].shape[-2] + token_embeddings.shape[1]) - self.context_len
            if diff > 0:
                #past_trim_tokens = task.tokens_per_timestep + action_tokens - 1
                for i in range(len(past_key_values)):
                    past_key_values[i] = past_key_values[i][:, :, :, diff:, :]
                    #past_key_values[i] = past_key_values[i][:, :, :, past_trim_tokens:, :]

        # predict tokens, sampling or deterministically picking best token
        #print('hey')
        for i in range(action_tokens):
            # if past_key_values is not None:
            #     print(past_key_values[0].shape)
            # print(token_embeddings.shape)
            #import pdb; pdb.set_trace()
            if past_key_values is not None:
                    for j in range(len(past_key_values)):
                        past_key_values[j] = past_key_values[j][:, :, :, -(self.context_len - 1):, :]

            logits, _, past_key_values = self.forward(
                token_embeddings=token_embeddings, 
                token_masks=token_masks, 
                token_target_masks=None, 
                tokens=None,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            # extract valid logits from last timestep
            logits = logits[0, -1, start_token:(end_token+1)]
            if deterministic:
                token = torch.argmax(logits, dim=-1)
            else:
                # sample from logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)[0]
            token = token + start_token

            # append to token_embeddings and token_masks
            token_masks = torch.cat([token_masks, torch.ones(token_masks.shape[0], 1, device=self.device)], dim=1)
            new_embedding = self.embed_token(token) # check shape of new_emebddingss
            token_embeddings = torch.cat([token_embeddings, new_embedding.reshape(1, 1, -1)], dim=1)
            # and trim to context len
            token_embeddings = token_embeddings[:, -self.context_len:, :]
            token_masks = token_masks[:, -self.context_len:]
            predicted_tokens.append(token)

            # trim off last token from past_key_values
            if past_key_values is not None: #and i != action_tokens - 1: 
                # only need the new
                token_embeddings = token_embeddings[:, -1:, :]
                token_masks = token_masks[:, -1:]
        past_key_values = past_key_values
        #import pdb; pdb.set_trace()
        # convert tokens back to actions
        if action_type == gym.spaces.Discrete:
            action = predicted_tokens[0] - start_token
        else:
            predicted_tokens = torch.stack(predicted_tokens, dim=0)
            action = self.continuous_action_tokenizer.decode(predicted_tokens)
        if past_key_values is not None:
            print(past_key_values[0].shape)
        return action, past_key_values

if __name__ == '__main__':
    model = GatoPolicy(
        device='cpu',
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

    #output = model(inputs)
    
    # Mix of image+discrete and continuous+continuous, and compute loss
    output = model([
        {
            'images': torch.randn(20, 3, 80, 64),
            'discrete_actions': torch.randint(0, 55, (20, 1)),
        },
        {
            'continuous_obs': torch.randn(15, 8),
            'continuous_actions': torch.randn(15, 4),
        }
    ], compute_loss=True)
