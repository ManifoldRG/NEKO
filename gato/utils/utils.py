import os
import json

import torch

class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def save_model(model, save_dir, save_name, args):
    # create save dir if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save args for loading model (if not already saved)
    args_path = os.path.join(save_dir, 'args.json')
    if not os.path.exists(args_path):
        with open(args_path, 'w') as f:
            json.dump(args, f)
    
    # save model
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(save_dir, save_name + '.pt'))
