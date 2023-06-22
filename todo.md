todo:
- verify/test ResNet patch embeddings
- add 1 channel image repeating in some place (in forward or when sampling from env?)
- prompting style during inference

things to add: 
- stochastic depth
- currently supporting Box and Discrete spaces, implement others: https://gymnasium.farama.org/api/spaces/

changes:
- added gradient clipping (can be disabled)

ideas:
- switch to more structured config instead of argparse, e.g. hydra. Want config files + flexibility to alter when running. 