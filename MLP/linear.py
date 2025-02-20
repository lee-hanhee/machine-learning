import torch.nn as nn

class Linear(nn.Module):
    '''
    - Parameters:
        - in_features: Number of input features (size of each input sample).
        - out_features: Number of output features (size of each output sample).
        - bias: If True, a learnable bias vector is added (default: True).
        - device: Specifies the device (CPU or GPU) where the layer's parameters should be stored (default: None, uses default device).
        - dtype: The data type of the layer’s parameters (default: None, uses default floating-point type).

    - Return:
        - output: A tensor of shape (batch_size, out_features) containing the transformed input.

    - Notes:
        - Computes a linear transformation of the input: `Y = XW^T + b`.
        - `W` is a learnable weight matrix of shape `(out_features, in_features)`.
        - `b` is a learnable bias vector of shape `(out_features)`, if `bias=True`.
        - The layer is commonly used in fully connected networks (MLPs), RNNs, and feature transformation layers.
    '''
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(x)
