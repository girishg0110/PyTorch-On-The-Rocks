import torch
from torch import nn
import numpy as np
from torch.autograd import grad

def getTorchFunction(forward_func, grad_func):
    class _TorchFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weights):
            ctx.save_for_backward(x, weights)
            result = forward_func(x, weights)
            result.requires_grad_(True)
            return result 
        
        @staticmethod
        def backward(ctx):
            x, weights = ctx.saved_tensors()
            return grad_func(x, weights)

    return _TorchFunction

class ClassicalLayer(nn.Module):
    def __init__(self, input_size, output_size, forward_func, grad_func, init_weights = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        weights = None
        if (init_weights is not None) and (init_weights.shape == torch.Size([output_size, input_size])):
            weights = init_weights
        else:
            weights = torch.zeros((output_size, input_size))
        self.weights = nn.Parameter(weights)
        self.function = getTorchFunction(forward_func, grad_func)

    def forward(self, x):
        return self.function.apply(x, *self.weights)