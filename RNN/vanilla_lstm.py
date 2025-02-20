import torch
import torch.nn as nn

class VanillaLSTMCell(nn.Module):
    '''
    Implements a basic LSTM cell with the following equations:

        f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)  # Forget gate
        i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)  # Input gate
        o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)  # Output gate
        c̃_t = tanh(W_c x_t + U_c h_{t-1} + b_c)    # Candidate cell state
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t            # Update cell state
        h_t = o_t ⊙ tanh(c_t)                      # Update hidden state

    - Parameters:
        - input_size: Number of features in x_t.
        - hidden_size: Number of hidden units in h_t and c_t.
    
    - Return:
        - h_t: Updated hidden state (batch_size, hidden_size).
        - c_t: Updated cell state (batch_size, hidden_size).
    '''
    
    def __init__(self, input_size, hidden_size):
        super(VanillaLSTMCell, self).__init__()

        self.hidden_size = hidden_size

        # Forget gate
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)

        # Input gate
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)

        # Output gate
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)

        # Candidate cell state
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_t, h_t_minus_1, c_t_minus_1):
        '''
        - Parameters:
            - x_t: Input at time step t (batch_size, input_size).
            - h_t_minus_1: Hidden state from previous time step (batch_size, hidden_size).
            - c_t_minus_1: Cell state from previous time step (batch_size, hidden_size).

        - Return:
            - h_t: Updated hidden state (batch_size, hidden_size).
            - c_t: Updated cell state (batch_size, hidden_size).
        '''
        
        # Compute LSTM gates
        f_t = self.sigmoid(self.W_f(x_t) + self.U_f(h_t_minus_1))  # Forget gate
        i_t = self.sigmoid(self.W_i(x_t) + self.U_i(h_t_minus_1))  # Input gate
        o_t = self.sigmoid(self.W_o(x_t) + self.U_o(h_t_minus_1))  # Output gate
        
        # Compute candidate cell state
        c_tilde = self.tanh(self.W_c(x_t) + self.U_c(h_t_minus_1))
        
        # Update cell state
        c_t = f_t * c_t_minus_1 + i_t * c_tilde
        
        # Update hidden state
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t  # Return updated hidden and cell states
