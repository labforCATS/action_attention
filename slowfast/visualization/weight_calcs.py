import torch
import numpy as np
import tqdm
import pdb

def get_model_weights(inputs = None, grads = None, activations = None, method = 'grad_cam'):
    if method == 'grad_cam':
        return(torch.mean(grads,dim=3))

    elif method == 'grad_cam_plusplus':
        grads = np.array(grads.cpu())
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(np.array(activations.cpu()), axis=(2,3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=3) # changed from (2,3)
        return(torch.FloatTensor(weights).cuda())
        
    elif method == 'eigen_cam':
        activations = activations.cpu().numpy().squeeze()
        activations[np.isnan(activations)] = 0
        # reshaped_activations = (activations).reshape(activations.shape[0], -1).transpose()
        reshaped_activations = (activations).reshape(activations.shape[-1], -1).transpose()
        # Centering before the SVD seems to be important here, otherwise the image returned is negative
        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[:-1])
        return(torch.FloatTensor(projection).cuda())
    
    elif method == 'score_cam':
        #TODO
        with torch.no_grad():
            pdb.set_trace()
            upsample = torch.nn.UpsamplingBilinear2d(size=inputs.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()
    
            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = inputs[:, None, :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16
            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights

    elif method == 'xgrad_cam':
        activations = np.array(activations.cpu())
        grads = np.array(grads.cpu())
        sum_activations = np.sum(activations, axis=3) # changed from (2,3)
        eps = 1e-7
        weights = grads * activations / (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2,3))
        return(torch.FloatTensor(weights).cuda())

    else:
        raise ValueError("invalid visualization method")
