import torch
from torch import nn
import numpy as np

class ClassicalLayer(nn.Module):
    def __init__(self, input_size, output_size, forward_func, grad_func, init_weights = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        weights = init_weights if (init_weights != None) else torch.zeros((output_size, input_size))
        self.weights = nn.Parameter(weights)

        class _TorchFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weights):
                ctx.save_for_later(x, weights)
                result = forward_func(x, weights)
                result.requires_grad_(True)
                return result 
            
            @staticmethod
            def backward(ctx):
                x, weights = ctx.saved_tensors()
                return grad_func(x, weights)

        self.forward_pass = _TorchFunction

    def forward(self, x):
        return self.forward_pass(x)