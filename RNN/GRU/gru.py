import torch
import torch.nn as nn

class GRU(nn.Module):
    '''
    - Parameters:
        - input_size: The number of expected features in the input `x`.
        - hidden_size: The number of features in the hidden state `h`.
        - num_layers: The number of recurrent layers (default: 1).
        - bias: If False, the layer does not use bias weights (default: True).
        - batch_first: If True, input and output tensors are provided as (batch, seq, feature) (default: False).
        - dropout: If nonzero, introduces a dropout layer on the outputs of each GRU layer except the last (default: 0.0).
        - bidirectional: If True, the GRU is bidirectional (default: False).
        - dtype: The data type for the parameters and buffers (default: None).

    - Return:
        - A tensor containing the output features from the last layer for each time step.
        - A tensor containing the final hidden state for each layer.

    - Notes:
        - Uses `torch.nn.GRU` to implement a gated recurrent unit (GRU) network.
        - When `bidirectional=True`, the output tensor will have twice the hidden_size.
        - The dropout value is only applied when `num_layers > 1`.
        - Unlike LSTM, GRU does not maintain a separate cell state, only a hidden state.
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 batch_first=False, dropout=0.0, bidirectional=False, dtype=None):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                          bias=bias, batch_first=batch_first, dropout=dropout, 
                          bidirectional=bidirectional, dtype=dtype)

    def forward(self, x, h0=None):
        return self.gru(x, h0)
