todo:
- verify/test ResNet patch embeddings (for Atari)
- for prompting style during inference
    - filter which episode to prompt with?
- make image padding configurable?

potential areas to look at for debugging results:
- verify token shifting, masking for during training step / calculating loss
- verify sampling from dataset
- verify transformers/gpt.py
- try disabling position encodings
- MuJoCo dataset are custom, could be a problem with dataset but unlikely. Could substitute in an official Minari dataset, e.g.: https://minari.farama.org/datasets/door/
- Check sampling and building episode from dataset

things to add: 
- stochastic depth
- currently supporting Box and Discrete spaces, implement others: https://gymnasium.farama.org/api/spaces/, like spaces.Text
- could improve tokenize_input_dicts() so that matching modality combinations are grouped before loop over dicts
- add/test geglu over gelu in transformer blocks

Purposeful changes from Gato:
- added gradient clipping (can be disabled)

ideas:
- switch to more structured config instead of argparse, e.g. hydra. Want config files + flexibility to alter when running. 
