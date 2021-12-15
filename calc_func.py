from classical_pytorch_layer import ClassicalLayer
import torch
from torch import nn
import numpy as np

# Linear (10 -> 1) Layer Example
input_dim = 1
output_dim = 1

def forward_sq(x, weights):
    return x*x

def grad_sq(x, weights):
    return -2*x

model = nn.Sequential(
    ClassicalLayer(input_dim, output_dim, forward_sq, grad_sq, init_weights = None)
)

data_point = torch.DoubleTensor([4]) # Data point
initial_val = model(data_point)
print(initial_val)

mse = initial_val
model.backwards()

print(data_point.grad())


