import torch
import torch.nn as nn

def mu_law(tensor, mu=100, M=256): 
    return torch.sign(tensor) * torch.log(1 + mu * torch.abs(tensor)) / torch.log(1 + mu*M)


class ContinuousTokenizer:
    def __init__(self, use_mu_law=True, mu=100, M=256, n_bins=1024, offset=None):
        self.use_mu_law = use_mu_law
        self.mu = mu
        self.M = M
        self.n_bins = n_bins
        self.offset = offset
    
    def encode(self, tensor):
        if self.use_mu_law:
            tensor = mu_law(tensor, self.mu, self.M)
        # clip to [-1, 1]
        tensor = torch.clamp(tensor, -1, 1)

        # discretize using uniform bins
        tensor = (tensor + 1) * (self.n_bins / 2)
        tensor = tensor.type(torch.int32)

        if self.offset is not None:
            tensor += self.offset

        return tensor
    
    def decode(self, tensor):
        if self.use_mu_law:
            raise Exception("mu-law encoding only expected with values which are not predicted")
        
        if self.offset is not None:
            tensor -= self.offset

        # convert back from discrete to continous values, discrete values should be from [0, 1023]
        tensor = (2 * tensor) / self.n_bins - 1

        return tensor




