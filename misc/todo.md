todo:
- Test Atari, make sure to verify/test ResNet patch embeddings (for Atari)
- test prompting style during inference
    - filter which episode to prompt with?
    - Test without prompting during inference, also during training
- make image padding configurable?
- 
things to add: 
- stochastic depth
- currently supporting Box and Discrete spaces, implement others: https://gymnasium.farama.org/api/spaces/, like spaces.Text
- could improve tokenize_input_dicts() so that matching modality combinations are grouped before loop over dicts
- add/test geglu over gelu in transformer blocks
- TransformerXL style memory during inference
- batched inference
- could try RoboCat style VQGAN image tokenization. Train VQGAN or test using a pretrained. 

Purposeful changes from Gato:
- added gradient clipping (can be disabled)

ideas:
- switch to a more structured config instead of argparse, e.g. hydra. Want config files + flexibility to alter when running. 
