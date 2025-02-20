import torch 
import torch.nn as nn

class LSTM(nn.Module):
    '''
    - Parameters:
        - input_size: The number of expected features in the input `x`.
        - hidden_size: The number of features in the hidden state `h` and cell state `c`.
        - num_layers: The number of recurrent layers (default: 1).
        - bias: If False, the layer does not use bias weights (default: True).
        - batch_first: If True, input and output tensors are provided as (batch, seq, feature) (default: False).
        - dropout: If nonzero, introduces a dropout layer on the outputs of each LSTM layer except the last (default: 0.0).
        - bidirectional: If True, the LSTM is bidirectional (default: False).
        - proj_size: If greater than 0, applies a projection layer to reduce the hidden state size (default: None).
        - dtype: The data type for the parameters and buffers (default: None).

    - Return:
        - output: A tensor containing the output features from the last layer for each time step.
        - (h_n, c_n): A tuple containing the final hidden state and cell state for each layer.

    - Notes:
        - Uses `torch.nn.LSTM` to implement a long short-term memory (LSTM) network.
        - When `bidirectional=True`, the output tensor will have twice the hidden_size.
        - The dropout value is only applied when `num_layers > 1`.
        - Unlike GRU, LSTM maintains both a hidden state (`h_n`) and a cell state (`c_n`).
        - The projection layer (`proj_size`) allows reducing hidden state size in deep LSTMs.
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 batch_first=False, dropout=0.0, bidirectional=False, proj_size=None, dtype=None):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            bias=bias, batch_first=batch_first, dropout=dropout, 
                            bidirectional=bidirectional, proj_size=proj_size, dtype=dtype)

    def forward(self, x, h0=None, c0=None):
        return self.lstm(x, (h0, c0) if h0 is not None and c0 is not None else None)
