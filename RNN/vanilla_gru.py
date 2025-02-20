import torch
import torch.nn as nn

class VanillaGRUCell(nn.Module):
    '''
    Implements a basic GRU cell following the update rules:

        z_t = sigmoid(W_z x_t + U_z h_{t-1})
        r_t = sigmoid(W_r x_t + U_r h_{t-1})
        h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}))
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

    - Parameters:
        - input_size: Number of features in x_t.
        - hidden_size: Number of hidden units in h_t.
    '''

    def __init__(self, input_size, hidden_size):
        super(VanillaGRUCell, self).__init__()

        # Linear layers for update gate (z_t)
            # Output must be of size hidden_size since z_t is used to update h_t
        self.W_z = nn.Linear(input_size, hidden_size) # self.W_z = W_z * x_t + b_z
        self.U_z = nn.Linear(hidden_size, hidden_size) # self.U_z = U_z * h_{t-1} + b_z

        # Linear layers for reset gate (r_t)
            # Output must be of size hidden_size since z_t is used to update h_t
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size)

        # Linear layers for candidate hidden state (h̃_t)
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_t, h_t_minus_1):
        '''
        - Parameters:
            - x_t: Input tensor at time step t (batch_size, input_size).
            - h_t_minus_1: Hidden state tensor from previous time step (batch_size, hidden_size).

        - Return:
            - h_t: Updated hidden state tensor at time step t (batch_size, hidden_size).
        '''
        # Compute update gate z_t
        z_t = self.sigmoid(self.W_z(x_t) + self.U_z(h_t_minus_1))

        # Compute reset gate r_t
        r_t = self.sigmoid(self.W_r(x_t) + self.U_r(h_t_minus_1))

        # Compute candidate hidden state h̃_t
        h_tilde = self.tanh(self.W_h(x_t) + self.U_h(r_t * h_t_minus_1))

        # Compute new hidden state h_t
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_tilde

        return h_t  # Only h_t is returned because GRUs don't output y_t explicitly
