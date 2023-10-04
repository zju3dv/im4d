import torch.nn as nn
activations = {
    'relu': nn.ReLU,
    'none': None
}

def get_activation(name):
    return activations[name]