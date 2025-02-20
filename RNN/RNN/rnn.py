import torch
import torch.nn as nn

class RNN(nn.Module):
    '''
    - Parameters:
        - input_size: Number of expected features in the input `x_t`.
        - hidden_size: Number of features in the hidden state `h_t`.
        - num_layers: Number of recurrent layers (default: 1).
        - nonlinearity: The activation function to use ('tanh' or 'relu'). Default is 'tanh'.
        - bias: If False, the layer does not use bias weights (default: True).
        - batch_first: If True, input and output tensors are provided as (batch, seq, feature) (default: False).
        - dropout: If nonzero, introduces a dropout layer on the outputs of each RNN layer except the last (default: 0.0).
        - bidirectional: If True, the RNN is bidirectional (default: False).

    - Return:
        - y_t+1: The predicted output tensor at the next time step.
        - h_t+1: The updated hidden state tensor at the next time step.

    - Notes:
        - Implements a simple RNN where the next output `y_{t+1}` is computed from `h_{t+1}`.
        - Uses `tanh` as the default activation function.
        - The hidden state is updated iteratively according to `h_t = tanh(W_h h_{t-1} + W_x x_t)`.
        - The output `y_{t+1}` is obtained from `MLP(h_{t+1})`.
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, 
                 batch_first=False, dropout=0.0, bidirectional=False):
        super(RNN, self).__init__()

        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, 
                          bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

        # Define a fully connected layer (MLP) to generate y_t+1 from h_t+1
        self.mlp = nn.Linear(hidden_size, input_size)  # Output size can be adjusted

    def forward(self, x_t, h_t):
        '''
        - Parameters:
            - x_t: The input tensor at time step `t`, shaped as (batch_size, input_size).
            - h_t: The hidden state tensor at time step `t`, shaped as (num_layers, batch_size, hidden_size).

        - Return:
            - y_t+1: The predicted output at the next time step, shaped as (batch_size, input_size).
            - h_t+1: The updated hidden state tensor for the next time step, shaped as (num_layers, batch_size, hidden_size).

        - Notes:
            - This function computes h_t+1 using the RNN equation and generates y_t+1 using an MLP.
        '''
        # Reshape x_t to match RNN input requirements: (seq_len=1, batch_size, input_size)
        x_t = x_t.unsqueeze(0)

        # Compute the next hidden state h_t+1
            # h_t_plus_1: hidden states for all time steps.
            # h_t_plus_1_final: final hidden state after processing all time steps.
        h_t_plus_1, h_t_plus_1_final = self.rnn(x_t, h_t)

        # Compute the next output y_t+1 using the MLP
        y_t_plus_1 = self.mlp(h_t_plus_1)

        # Remove the sequence dimension (squeeze the first dimension)
        y_t_plus_1 = y_t_plus_1.squeeze(0)

        return y_t_plus_1, h_t_plus_1_final  # Return y_{t+1} and h_{t+1}

if __name__ == '__main__':
    # Example usage:
    input_size = 10   # Number of input features (x_t dimension)
    hidden_size = 20  # Number of hidden units (h_t dimension)
    batch_size = 3    # Number of sequences per batch

    # Create the RNN model
    model = RNN(input_size, hidden_size)

    # Initialize input and hidden state
    x_t = torch.randn(batch_size, input_size)  # Shape: (batch_size, input_size)
    h_t = torch.randn(1, batch_size, hidden_size)  # Shape: (num_layers=1, batch_size, hidden_size)

    # Compute next output and hidden state
    y_t_plus_1, h_t_plus_1 = model(x_t, h_t)

    # Print results
    print("y_{t+1} shape:", y_t_plus_1.shape)  # Should be (batch_size, input_size)
    print("h_{t+1} shape:", h_t_plus_1.shape)  # Should be (num_layers, batch_size, hidden_size)
