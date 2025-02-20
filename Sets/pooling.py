import torch 

def sum_pool(data):
    '''
    - Parameters:
        - data: A PyTorch tensor containing numerical values.
    - Return:
        - A tensor representing the sum of data along dimension 0.
    - Notes:
        - Uses torch.sum() to compute the sum along the specified axis.
    '''
    return torch.sum(data, dim=0)

def mean_pool(data, keepdim=False):
    '''
    - Parameters:
        - data: A PyTorch tensor containing numerical values.
        - keepdim: A boolean flag indicating whether to retain the reduced dimension (default: False).
    - Return:
        - A tensor representing the mean of data along dimension 0.
    - Notes:
        - Uses torch.mean() to compute the mean along the specified axis.
    '''
    return torch.mean(data, dim=0, keepdim=keepdim)

def var_pool(data, unbiased=True):
    '''
    - Parameters:
        - data: A PyTorch tensor containing numerical values.
        - unbiased: A boolean flag indicating whether to use the unbiased estimator (default: True).
    - Return:
        - A tensor representing the variance of data along dimension 0.
    - Notes:
        - Uses torch.var() to compute the variance along the specified axis.
    '''
    return torch.var(data, dim=0, unbiased=unbiased)

def std_pool(data, unbiased=True):
    '''
    - Parameters:
        - data: A PyTorch tensor containing numerical values.
        - unbiased: A boolean flag indicating whether to use the unbiased estimator (default: True).
    - Return:
        - A tensor representing the standard deviation of data along dimension 0.
    - Notes:
        - Uses torch.std() to compute the standard deviation along the specified axis.
    '''
    return torch.std(data, dim=0, unbiased=unbiased)

def max_pool(data):
    '''
    - Parameters:
        - data: A PyTorch tensor containing numerical values.
    - Return:
        - A tensor containing the maximum values along dimension 0.
    - Notes:
        - Uses torch.max() to find the maximum along the specified axis.
    '''
    return torch.max(data, dim=0)

def min_pool(data):
    '''
    - Parameters:
        - data: A PyTorch tensor containing numerical values.
    - Return:
        - A tensor containing the minimum values along dimension 0.
    - Notes:
        - Uses torch.min() to find the minimum along the specified axis.
    '''
    return torch.min(data, dim=0)

def count(data):
    '''
    - Parameters:
        - data: A PyTorch tensor containing numerical values.
    - Return:
        - An integer representing the number of elements along dimension 0.
    - Notes:
        - Uses data.shape[0] or data.size(0) to determine the count.
    '''
    return data.shape[0]  # or data.size(0)
