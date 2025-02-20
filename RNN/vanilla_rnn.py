import torch
import torch.nn as nn

class VanillaRNNCell(nn.Module):
    '''
    Implements a basic RNN cell that follows the update rule:
    
        h_t = tanh(W_h h_{t-1} + W_x x_t)
        y_t = MLP(h_t)
    
    - Parameters:
        - input_size: Number of features in x_t.
        - hidden_size: Number of hidden units in h_t.
        - output_size: Number of features in y_t.
    '''

    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNNCell, self).__init__()

        # Linear transformations for x_t and h_{t-1}
        self.W_x = nn.Linear(input_size, hidden_size)   # Linear(x_t)
        self.W_h = nn.Linear(hidden_size, hidden_size)  # Linear(h_{t-1})

        # Activation function (tanh for hidden state update)
        self.activation = nn.Tanh()

        # Output MLP (Linear layer to compute y_t)
        self.mlp = nn.Linear(hidden_size, output_size)

    def forward(self, x_t, h_t_minus_1):
        '''
        - Parameters:
            - x_t: Input tensor at time step t (batch_size, input_size).
            - h_t_minus_1: Hidden state tensor from previous time step (batch_size, hidden_size).

        - Return:
            - y_t: Output tensor at time step t (batch_size, output_size).
            - h_t: Updated hidden state tensor at time step t (batch_size, hidden_size).
        '''
        # Compute the new hidden state h_t
        h_t = self.activation(self.W_x(x_t) + self.W_h(h_t_minus_1))

        # Compute the output y_t using MLP
        y_t = self.mlp(h_t)

        return y_t, h_t  # Return output and updated hidden state
