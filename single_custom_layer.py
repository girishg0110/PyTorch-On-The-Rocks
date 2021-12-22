import torch
from torch import nn
import numpy as np

class single_custom_layer(nn.Module):
    """
    This function defines a single custom torch layer.
    """
    def __init__(self, in_features: int, out_features: int, weights:int, agf:torch.autograd.Function):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.zeros(weights))

        self.agf = agf

    def reset_parameters(self) -> None:
        pass

    def forward(self, input):
        return self.agf.apply(input, self.weights)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

def quantum_circuit():

    class qc_torch_func(torch.autograd.Function):
        """

        """

        @staticmethod
        def forward(ctx, input, weights):
            """

            """
            ctx.save_for_backward(input, weights)
            return input

        @staticmethod
        def backward(ctx, grad_outputs):
            """

            """
            input, weights = ctx.saved_tensors
            return grad_outputs

    return qc_torch_func


class qnn_map():
    """

    """

if __name__ == "__main__":
    """

    """
    qc = quantum_circuit()
    model1 = single_custom_layer(10, 10, 1, qc)
    model2 = single_custom_layer(10, 5, 1, qc)
    seq_model = nn.Sequential(model1, model2)
    print(seq_model)

    data_point = torch.tensor(np.ones(10))
    print(seq_model(data_point))