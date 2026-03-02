import torch
import math
from torch.optim.optimizer import Optimizer


class UniSAGEOptimized(Optimizer):
    """
    SAGE optimized for minimal CPU overhead required to dispatch for each loop.
    Separated step operation branching so the operations can easily be dispatched in parallel.

    Key Implementation Details:
     - Bucketing: In every step, we loop through params once to sort them into sage_1d, sage_2d, and sinkgd lists.

     - Foreach Math: We replace loops like p.mul_() with torch._foreach_mul_(params, ...) where possible.

     - Handling Broadcasting: torch._foreach_ functions typically do not support broadcasting
       (e.g., multiplying a [N, D] tensor by a [1, D] tensor). Therefore, for SAGE-2D, we split the logic:
         we use foreach for the state updates (which match shapes) but fall back to a simple loop for the final broadcasted multiplication.
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
        embedding_heuristic=True
    ):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
            sinkhorn_iter=sinkhorn_iter, sinkhorn_scale=sinkhorn_scale,
            embedding_heuristic=embedding_heuristic
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Bucket Parameters
            sage_1d_lists = {'params': [], 'grads': [], 'exp_avgs': [], 's_ranges': [], 'steps': []}
            sage_2d_lists = {'params': [], 'grads': [], 'exp_avgs': [], 's_ranges': [], 'steps': []}
            sinkgd_lists  = {'params': [], 'grads': []}
            
            param_names = group.get('name', None)
            
            for p in group['params']:
                if p.grad is None: continue
                if p.grad.is_sparse: raise RuntimeError('UniSAGE does not support sparse gradients')

                # Determine strategy
                use_sinkhorn = False
                if p.ndim == 2:
                    use_sinkhorn = True
                    rows, cols = p.shape
                    
                    # Heuristic checks
                    is_embedding = False
                    if group['embedding_heuristic'] and rows > 50000 and rows > cols * 10: is_embedding = True
                    if param_names and 'embed' in param_names: is_embedding = True
                    if hasattr(p, 'is_embedding') and p.is_embedding: is_embedding = True
                    
                    if is_embedding: use_sinkhorn = False

                if use_sinkhorn:
                    sinkgd_lists['params'].append(p)
                    sinkgd_lists['grads'].append(p.grad)
                else:
                    # SAGE (Stateful)
                    state = self.state[p]
                    
                    # Lazy initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if p.ndim > 1: # 2D SAGE (Embedding)
                            reduced_shape = p.mean(dim=0, keepdim=True)
                            state['s_range'] = torch.zeros_like(reduced_shape, memory_format=torch.preserve_format)
                        else: # 1D SAGE (Bias/Norm)
                            state['s_range'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1
                    
                    # Sort into 1D vs 2D buckets
                    target_list = sage_2d_lists if p.ndim > 1 else sage_1d_lists
                    target_list['params'].append(p)
                    target_list['grads'].append(p.grad)
                    target_list['exp_avgs'].append(state['exp_avg'])
                    target_list['s_ranges'].append(state['s_range'])
                    target_list['steps'].append(state['step'])

            # Execute Foreach Updates
            if sage_1d_lists['params']:
                self._sage_foreach_step(sage_1d_lists, group, is_2d=False)
            
            if sage_2d_lists['params']:
                self._sage_foreach_step(sage_2d_lists, group, is_2d=True)
            
            if sinkgd_lists['params']:
                self._sinkgd_step(sinkgd_lists, group)

        return loss

    def _sage_foreach_step(self, lists, group, is_2d):
        params, grads = lists['params'], lists['grads']
        exp_avgs, s_ranges = lists['exp_avgs'], lists['s_ranges']
        steps = lists['steps']
        
        beta1, beta2 = group['betas']
        lr = group['lr']
        wd = group['weight_decay']
        eps = group['eps']

        # Global Weight Decay
        if wd != 0:
            torch._foreach_mul_(params, 1 - lr * wd)

        # S-Range Calculation
        if is_2d:
            # SAGE-2D: Reduce [N, D] -> [1, D]
            s_t_list = [g.abs().mean(dim=0, keepdim=True) for g in grads]
        else:
            # SAGE-1D: Element-wise abs [D] -> [D] (Foreach valid but abs is fast)
            s_t_list = [g.abs() for g in grads]

        # Update S-Range EMA (Foreach)
        # s_range = s_range * beta2 + s_t * (1-beta2)
        torch._foreach_mul_(s_ranges, beta2)
        torch._foreach_add_(s_ranges, s_t_list, alpha=1.0 - beta2)

        # Calculate Bias-Corrected Scaling
        bias_correction2 = [1.0 - beta2 ** s for s in steps]
        
        s_range_corrected = torch._foreach_div(s_ranges, bias_correction2)

        # Relative Damping
        s_rms_list = [torch.sqrt(src.square().mean()) for src in s_range_corrected]
        
        denoms = torch._foreach_add(s_range_corrected, eps)
        
        raw_dampers = []
        for rms, denom in zip(s_rms_list, denoms):
            raw_dampers.append(rms / denom)

        # step_scale = clamp(raw, max=1.0)
        # _foreach_clamp_max_ is available in newer PyTorch, else use loop
        if hasattr(torch, "_foreach_clamp_max"):
            step_scales = torch._foreach_clamp_max(raw_dampers, 1.0)
        else:
            step_scales = [d.clamp(max=1.0) for d in raw_dampers]

        # Update Momentum
        torch._foreach_mul_(exp_avgs, beta2)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta2)

        # Calculate Update Direction
        updates = torch._foreach_mul(exp_avgs, beta1)
        torch._foreach_add_(updates, grads, alpha=1.0 - beta1)
        
        # Sign
        for u in updates: u.sign_()

        # Apply Damping & Step
        if is_2d:
            for u, s in zip(updates, step_scales):
                u.mul_(s)
        else:
            torch._foreach_mul_(updates, step_scales)

        # Final Step
        torch._foreach_add_(params, updates, alpha=-lr)

    def _sinkgd_step(self, lists, group):
        params, grads = lists['params'], lists['grads']
        lr = group['lr']
        wd = group['weight_decay']
        L = group['sinkhorn_iter']
        sink_scale = group['sinkhorn_scale']
        eps = group['eps']

        for p, g in zip(params, grads):
            if wd != 0:
                p.data.mul_(1 - lr * wd)
            
            update = g.clone()
            
            for _ in range(L):
                row_norm = torch.linalg.vector_norm(update, dim=1, keepdim=True)
                update.div_(row_norm.add_(eps))
                col_norm = torch.linalg.vector_norm(update, dim=0, keepdim=True)
                update.div_(col_norm.add_(eps))
            
            p.add_(update, alpha=-(lr * sink_scale))
