import argparse
from utils import DotDict

def main(args):
    print(args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input & tokenization
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--patch_size', type=int, default=16) # image patch size
    parser.add_argument('--sequence_length', type=int, default=1024) # number of tokens in seq

    parser.add_argument('--vocab_size', type=int, default=32000) # number of tokens from SentencePiece
    parser.add_argument('--continuous_tokens', type=int, default=1024) # number of tokens for continuous values (e.g. actions, observations)
    parser.add_argument('--discrete_tokens', type=int, default=1024) # number of discrete action tokens

    # transformer hyperparameters
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--heads', type=int, default=24)
    parser.add_argument('--embed_dim', type=int, default=768)


    # training
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--datasets', type=str, nargs='+', default=['d4rl_halfcheetah-expert-v2'])
    
    

    args = parser.parse_args()
    args = DotDict(vars(args))
    main(args)