from torch import nn

class Host:
    def __init__(self, neural_network: nn.Module):
        self.neural_network = neural_network
    
    def inital_training(self):
        pass
    
    def federated_training(self):
        pass