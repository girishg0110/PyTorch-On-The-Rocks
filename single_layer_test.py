from classical_pytorch_layer import ClassicalLayer
import torch
from torch import nn
import numpy as np

# Linear (10 -> 1) Layer Example
input_dim = 10
output_dim = 1

def forward_linear(x, weights):
    return (weights @ x.T)

def grad_linear(x, weights):
    return weights

model = nn.Sequential(
    ClassicalLayer(input_dim, output_dim, forward_linear, grad_linear, init_weights = torch.DoubleTensor([range(9, -1, -1)]))
)

data_point = torch.DoubleTensor([range(10)]) # Data point
initial_val = model(data_point)
print(initial_val)
