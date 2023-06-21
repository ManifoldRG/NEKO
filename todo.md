todo:
- verify/test ResNet patch embeddings
- how to input grayscale (1 channel) input? (answer: repeat)

implentation features:
- stochastic depth
- cosine lr schedule
- currently supporting Box and Discrete spaces, implement others: https://gymnasium.farama.org/api/spaces/

ideas:
- switch to more structured config instead of argparse, e.g. hydra. Want config files + flexibility to alter when running. 