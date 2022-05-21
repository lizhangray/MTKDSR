import os
import torch
import torch.nn as nn
import pdb


class BaseModel():
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_train = False
        self.schedulers = []
        self.optimizers = []

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        pretrained_dict = torch.load(load_path)
        model_dict = network.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)
        
