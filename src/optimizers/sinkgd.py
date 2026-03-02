import torch
import math
from torch.optim.optimizer import Optimizer

class SinkGD(Optimizer):
    """
    Implements the SinkGD (Sinkhorn Gradient Descent) optimizer proposed in
    'Gradient Multi-Normalization for Stateless and Scalable LLM Training' (Scetbon et al., 2025).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): global learning rate (default: 1e-3).
        sinkhorn_iter (int, optional): number of normalization iterations (L) for matrices (default: 5).
        linear_lr_scale (float, optional): scaling factor (alpha) for 2D/linear parameters (default: 0.05).
        betas (Tuple[float, float], optional): Adam coefficients (default: (0.9, 0.999)).
        eps (float, optional): numerical stability term (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.01).
        embedding_heuristic (bool, optional): If True, attempts to identify embedding layers 
            by shape characteristics (vocab_size usually >> hidden_dim) if names aren't available.
            Default: False (treats all 2D tensors as SinkGD layers).
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        sinkhorn_iter: int = 5,
        linear_lr_scale: float = 5e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        embedding_heuristic: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if sinkhorn_iter < 1:
            raise ValueError(f"Invalid sinkhorn_iter: {sinkhorn_iter}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            sinkhorn_iter=sinkhorn_iter,
            linear_lr_scale=linear_lr_scale,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            embedding_heuristic=embedding_heuristic
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            
            # Extra metadata if provided by user (e.g. via custom param groups)
            param_names = group.get('name', None) 

            for idx, p in enumerate(group['params']):
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('SinkGD does not support sparse gradients')
                    grads.append(p.grad)
                    
                    is_sinkhorn_eligible = (p.ndim > 1)
                    
                    # Check explicit name exclusion
                    if is_sinkhorn_eligible and param_names:
                        if 'embed' in param_names: 
                            is_sinkhorn_eligible = False

                    # 2. Check heuristic exclusion (useful for Transformers where Vocab >> Hidden)
                    if is_sinkhorn_eligible and group['embedding_heuristic']:
                        rows, cols = p.shape
                        # Heuristic: If one dim is significantly larger than the other (often vocab size),
                        # and it's the first dim (PyTorch Embedding is num_embeddings x embedding_dim),
                        # it might be an embedding.
                        if rows > 50000 and rows > cols * 10:
                             is_sinkhorn_eligible = False

                    # Fallback to Adam state
                    if not is_sinkhorn_eligible:
                        state = self.state[p]
                        if len(state) == 0:
                            state['step'] = 0
                            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])
                        state_steps.append(state['step'])
                    else:
                        exp_avgs.append(None)
                        exp_avg_sqs.append(None)
                        state_steps.append(None)
            
            self._sinkgd_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                group
            )

        return loss

    def _sinkgd_step(self, params, grads, exp_avgs, exp_avg_sqs, state_steps, group):
        lr = group['lr']
        weight_decay = group['weight_decay']
        sinkhorn_iter = group['sinkhorn_iter']
        linear_scale = group['linear_lr_scale']
        beta1, beta2 = group['betas']
        eps = group['eps']

        for i, param in enumerate(params):
            grad = grads[i]

            is_adam_param = exp_avgs[i] is not None

            # Weight Decay
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            if not is_adam_param:
                # SinkGD (Stateless)
                X = grad.clone()
                m, n = X.shape
                
                sqrt_n = math.sqrt(n)
                sqrt_m = math.sqrt(m)

                for _ in range(sinkhorn_iter):
                    row_norms = torch.linalg.vector_norm(X, dim=1, keepdim=True)
                    row_norms.add_(eps) 
                    X.div_(row_norms).mul_(sqrt_n)

                    col_norms = torch.linalg.vector_norm(X, dim=0, keepdim=True)
                    col_norms.add_(eps)
                    X.div_(col_norms).mul_(sqrt_m)
                
                effective_lr = lr * linear_scale
                param.add_(X, alpha=-effective_lr)

            else:
                # AdamW (Stateful)
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i] + 1
                
                self.state[param]['step'] = step

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)
                step_size = lr / (1 - beta1 ** step)
                
                param.addcdiv_(exp_avg, denom, value=-step_size)
