import torch

def mlp_inference(x, weights, biases, hidden_act_fn, output_act_fn=None):
    '''
    - Parameters:
        - x: Input tensor of shape (batch_size, input_size).
        - weights: List of weight matrices (each of shape appropriate to MLP layers).
        - biases: List of bias vectors corresponding to each weight matrix.
        - hidden_act_fn: Activation function for hidden layers.
        - output_act_fn: Optional activation function for the output layer (default: None).
    
    - Return:
        - h: Output tensor after forward propagation through the MLP.
    
    - Notes:
        - Applies `hidden_act_fn` to all layers **except the final layer**.
        - If `output_act_fn` is provided, it is applied to the final layer.
    '''

    h = x
    num_layers = len(weights)
    
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = torch.matmul(h, W) + b  # Linear transformation

        # Apply activation only to hidden layers (not the final layer)
        if i < num_layers - 1:
            h = hidden_act_fn(h)
    
    # Apply output activation function if specified
    if output_act_fn is not None:
        h = output_act_fn(h)

    return h
