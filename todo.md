todo:
- verify/test ResNet patch embeddings
- for prompting style during inference
    - filter which episode to prompt with?
- make image padding configurable?

things to add: 
- stochastic depth
- currently supporting Box and Discrete spaces, implement others: https://gymnasium.farama.org/api/spaces/, like spaces.Text

Purposeful changes from Gato:
- added gradient clipping (can be disabled)

ideas:
- switch to more structured config instead of argparse, e.g. hydra. Want config files + flexibility to alter when running. 