import torch
import math
from torch.optim.optimizer import Optimizer

class UniSAGE(Optimizer):
    """
    SAGE: A hybrid optimizer using:
    1. SAGE for the embedding layer: Lion-like sign update with a low-memory O(d) 
       per-dimension/channel adaptive scale. (Memory-Efficiency)
    2. Sinkhorn normalization for other 2D dense layers. (Stateless)
    """
    def __init__(
        self, 
        params, 
        lr=2e-3, 
        betas=(0.9, 0.99), 
        eps=1e-8,
        weight_decay=1e-2, 
        sinkhorn_iter=5, 
        sinkhorn_scale=10, 
        hybrid=True,
        embedding_heuristic=True
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
            sinkhorn_iter=sinkhorn_iter, sinkhorn_scale=sinkhorn_scale,
            hybrid=hybrid, embedding_heuristic=embedding_heuristic
        )
        super(UniSAGE, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps, wd = group['eps'], group['weight_decay']
            sink_iter, sink_scale = group['sinkhorn_iter'], group['sinkhorn_scale']
            
            param_names = group.get('name', None)

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('UniSAGE does not support sparse gradients')

                # 1. Decide Strategy
                use_sinkhorn = False
                if group['hybrid'] and p.ndim == 2:
                    use_sinkhorn = True
                    
                    # Check if it is an Embedding
                    rows, cols = p.shape
                    is_embedding = False
                    
                    if group['embedding_heuristic'] and rows > 50_000 and rows > cols * 10: 
                        is_embedding = True
                    if param_names and 'embed' in param_names:
                        is_embedding = True
                    if hasattr(p, 'is_embedding') and p.is_embedding:
                        is_embedding = True
                    
                    if is_embedding:
                        use_sinkhorn = False

                # 2. Execute Step
                if use_sinkhorn:
                    self._sinkgd_single_step(p, lr, wd, sink_iter, sink_scale, eps)
                else:
                    self._sage_single_step(p, lr, wd, beta1, beta2, eps)

        return loss

    def _sinkgd_single_step(self, p, lr, wd, L, scale, eps):
        grad = p.grad
        
        # Weight Decay
        if wd != 0:
            p.data.mul_(1 - lr * wd)

        # Uniform-Sinkhorn Normalization
        update = grad.clone()
        
        for _ in range(L):
            row_norm = torch.linalg.vector_norm(update, dim=1, keepdim=True)
            update.div_(row_norm.add_(eps))
            col_norm = torch.linalg.vector_norm(update, dim=0, keepdim=True)
            update.div_(col_norm.add_(eps))
        
        p.add_(update, alpha=-(lr * scale))

    def _sage_single_step(self, p, lr, wd, beta1, beta2, eps):
        grad = p.grad
        state = self.state[p]

        # Initialization
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            if p.ndim > 1:
                reduced_shape = p.mean(dim=0, keepdim=True)
                state['s_range'] = torch.zeros_like(reduced_shape, memory_format=torch.preserve_format)
            else:
                state['s_range'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        state['step'] += 1
        exp_avg = state['exp_avg']
        s_range = state['s_range']

        # Weight Decay
        if wd != 0:
            p.data.mul_(1 - lr * wd)

        # 1. S-Range(S_t) Update
        grad_abs = grad.abs()
        if p.ndim > 1:
            s_t = grad_abs.mean(dim=0, keepdim=True)
        else:
            s_t = grad_abs

        s_range.mul_(beta2).add_(s_t, alpha=1.0 - beta2)
        bias_correction2 = 1.0 - beta2 ** state['step']
        s_range_corrected = s_range / bias_correction2

        # 2. RMS-Relative Damping
        s_mean_sq = torch.mean(s_range_corrected.square())
        s_rms = torch.sqrt(s_mean_sq)

        raw_damper = s_rms / (s_range_corrected + eps)
        step_scale = torch.clamp(raw_damper, max=1.0)

        # -- Fast-Fuse for preventing lagged damping
        s_t_mean_sq = torch.mean(s_t.square())
        s_t_rms = torch.sqrt(s_t_mean_sq)
        instant_damper = s_t_rms / (s_t + eps)
        final_scale = torch.min(step_scale, instant_damper)

        # 3. Final updates
        exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        
        update = exp_avg.clone().mul_(beta1).add(grad, alpha=1.0 - beta1).sign_()
        update.mul_(final_scale)

        p.add_(update, alpha=-lr)
