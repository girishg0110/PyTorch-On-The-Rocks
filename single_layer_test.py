from classical_pytorch_layer import ClassicalLayer
import torch
from torch import nn
import numpy as np

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
print(model)

data_point = torch.tensor(np.ones(10)) # Data point
target_score = torch.tensor(7) # Target output of network on data_point

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