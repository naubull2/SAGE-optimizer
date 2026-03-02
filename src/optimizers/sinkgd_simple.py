import torch
import math
from collections import ChainMap
from torch.optim.optimizer import Optimizer
from typing import Iterable, List, Dict


class SinkGD(Optimizer):
    def __init__(self, params, lr=2e-2, L=5, eps=1e-8, sinkhorn_scale=5e-2, **kwargs):
        """
        Simplified implementation of the SinkGD optimizer.
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            L (int): Number of alternating row/column normalizations.
            eps (float): Small value to prevent division by zero.
        """
        lr *= sinkhorn_scale
        defaults = dict(lr=lr, L=L, eps=eps)
        super(SinkGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            iterations = group['L']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                
                # We only apply Sinkhorn to 2D matrices (weights of Linear layers)
                if grad.dim() == 2:
                    # Make a copy to avoid in-place modification of the original grad
                    normalized_grad = grad.clone()

                    rows, cols = normalized_grad.shape
                    for _ in range(iterations):
                        # Row normalization with sqrt(cols) scaling factor
                        row_norm = torch.linalg.norm(normalized_grad, ord=2, dim=1, keepdim=True)
                        normalized_grad = math.sqrt(cols) * normalized_grad / row_norm.add_(eps)
                        # Column normalization with sqrt(rows) scaling factor
                        col_norm = torch.linalg.norm(normalized_grad, ord=2, dim=0, keepdim=True)
                        normalized_grad = math.sqrt(rows) * normalized_grad / col_norm.add_(eps)
                    
                    update_tensor = normalized_grad
                else:
                    # For non-matrix params (biases, etc.), use a simple SGD-style update
                    # or just normalize the vector.
                    update_tensor = grad / (torch.linalg.norm(grad) + eps)

                # Apply the final update
                p.add_(update_tensor, alpha=-lr)
                
        return loss
