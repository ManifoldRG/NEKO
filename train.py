import argparse
from utils import DotDict

def main(args):
    print(args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input & tokenization
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--patch_size', type=int, default=16) # image patch size
    parser.add_argument('--sequence_length', type=int, default=1024) # number of tokens in seq

    parser.add_argument('--vocab_size', type=int, default=32000) # number of tokens from SentencePiece
    parser.add_argument('--action_size', type=int, default=1024)
    parser.add_argument('--continuous_values_size', type=int, default=1024)

    # transformer hyperparameters
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--heads', type=int, default=24)
    parser.add_argument('--embed_dim', type=int, default=768)


    # training
    parser.add_argument('--dropout', type=float, default=0.1)
    

    args = parser.parse_args()
    args = DotDict(vars(args))
    main(args)