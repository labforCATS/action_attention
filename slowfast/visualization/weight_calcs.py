import torch
import numpy as np
import tqdm
import pdb

def get_model_weights(inputs = None, grads = None, activations = None, method = 'grad_cam'):
    # pdb.set_trace()
    if method == 'grad_cam':
        return(torch.mean(grads,dim=3))

    elif method == 'grad_cam_plusplus':
        grads_power_2 = torch.pow(grads, 2)
        grads_power_3 = torch.pow(grads, 3)
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = torch.sum(activations, axis=(2,3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:,:,None,None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = torch.where(grads != 0, aij, 0)
        weights = torch.maximum(grads, torch.zeros_like(grads)) * aij
        weights = torch.sum(weights, axis=3)
        return(weights)
        
    elif method == 'eigen_cam':
        activations = activations.squeeze()
        activations[torch.isnan(activations)] = 0
        reshaped_activations = (activations).reshape(activations.shape[-1], -1).transpose(0, 1)
        # reshaped_activations = (new_activations).reshape(new_activations.shape[-1], -1).transpose(0,1)
        # Centering before the SVD seems to be important here, otherwise the image returned is negative
        reshaped_activations = reshaped_activations - torch.mean(reshaped_activations, axis=0)
        U, S, VT = torch.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0,:]
        projection = projection.reshape(activations.shape[:-1])
        return(projection)

    elif method == 'xgrad_cam':
        sum_activations = torch.sum(activations, axis=3)
        eps = 1e-7
        weights = grads * activations / (sum_activations[:,:,None,None] + eps)
        weights = weights.sum(axis=(2,3))
        return(weights)

    else:
        raise ValueError("invalid visualization method")
