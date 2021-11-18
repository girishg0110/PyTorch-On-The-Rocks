import torch
from torch import nn
import numpy as np

class ClassicalLayer(nn.Module):
    def __init__(self, input_size, output_size, init_weights, forward_func, grad_func):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = init_weights

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

# Linear (10 -> 1) Layer Example
input_dim = 10
output_dim = 1

def forward_linear(x, weights):
    return (x @ weights.T).item()

def grad_linear(x, weights):
    return -weights

model = nn.Sequential(
    ClassicalLayer(input_dim, output_dim, forward_linear, grad_linear)
)

data_point = torch.tensor(np.ones(10)) # Data point
target_score = torch.tensor(7). # Target output of network on data_point

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crit = torch.nn.CrossEntropyLoss()
opt.zero_grad()
predicted_val = model(data_point)
print("{0} is the initial score. Should be 0.".format(predicted_val))

loss = crit(predicted_val, target_score)
loss.backward()
opt.step()

predicted_val = model(data_point)
print("{0} is the score after one pass. Should be neq 0.".format(predicted_val))