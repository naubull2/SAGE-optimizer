import torch
import math
from torch.optim.optimizer import Optimizer


class SAGE(Optimizer):
    """
    SAGE: A hybrid optimizer using SR-opt for embeddings, Sinkhorn for 2D layers, and AdamW for 1D.
    This class is not the final SAGE from the paper, but more of a experimentation code for exploring various options.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=1e-2, sinkhorn_iterations=5, sinkhorn_scale=20,
                 schedule_type=None, hybrid=False, lion=False, tied_embedding=True, **kwargs):

        if schedule_type not in ['log', 'sqrt', None]:
            raise ValueError(f"Invalid schedule_type: {schedule_type}. Must be 'log' or 'sqrt'.")
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
            
        self.lion = lion
        self.hybrid = hybrid
        self.sinkhorn_scale = sinkhorn_scale
        self.tied_embedding = tied_embedding
        defaults = dict(lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, L=sinkhorn_iterations,
                         schedule_type=schedule_type)
        super(SAGE, self).__init__(params, defaults)


    def _calculate_schedule_factor(self, step, schedule_type):
        """Calculates the dynamic learning rate decay factor."""
        step_float = float(step)
        if schedule_type == 'sqrt':
            return 1.0 / math.sqrt(step_float)
        elif schedule_type == 'log':
            return 1.0 / math.log(step_float + 1.0) 
        return 1.0


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, beta1, beta2, eps, wd, L = group['lr'], group['betas'][0], group['betas'][1], group['eps'], group['weight_decay'], group['L']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                param_name = group.get('name', '')
                # Check for embedding layer
                is_embedding = 'embed' in param_name.lower() or (hasattr(p, 'is_embedding') and p.is_embedding)

                if p.dim() == 2 and (is_embedding or not self.hybrid):
                    # SR-Opt Logic for Embedding Layer
                    state = self.state[p]
                    if self.lion:
                        # Init state
                        if len(state) == 0:
                            state['exp_avg'] = torch.zeros_like(p)

                        exp_avg = state['exp_avg']
                        # Weight decay
                        p.data.mul_(1. - lr * wd)

                        # Weight update
                        update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1. - beta1).sign_()
                        p.add_(update, alpha = -lr)

                        # Decay momentum
                        exp_avg.mul_(beta2).add_(grad, alpha = 1. - beta2)

                    else:
                        if len(state) == 0:
                            # Momentum state
                            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            
                            # Soft-Range Scale state
                            state['s_range'] = torch.empty_like(p.narrow(0, 0, 1), memory_format=torch.preserve_format).zero_()
                            state['step'] = 0
                        state['step'] += 1
                        exp_avg = state['exp_avg']
                        s_range = state['s_range']

                        # Weight decay
                        p.data.mul_(1. - lr * wd)

                        # S-Range Calculation
                        grad_abs = grad.abs()
                        s_t = grad_abs.mean(dim=0, keepdim=True)

                        s_range.mul_(beta2).add_(s_t, alpha=1.0 - beta2)

                        bias_correction2 = 1.0 - beta2 ** state['step']
                        s_range_corrected = s_range / bias_correction2

                        # Calculate Layer-wise RMS
                        s_sq = s_range_corrected.square()
                        s_mean_sq = torch.mean(s_sq)
                        s_rms = torch.sqrt(s_mean_sq)

                        # Calculate Relative Damper
                        raw_damper = s_rms / (s_range_corrected + eps)
                        step_scale = torch.clamp(raw_damper, max=1.0)
                        state['step_scale'] = step_scale
                        
                        # Weight update
                        update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1. - beta1).sign_()
                        update = update.mul(step_scale)

                        # Final Parameter Update
                        p.add_(update, alpha=-lr)

                        # Decay momentum
                        exp_avg.mul_(beta2).add_(grad, alpha = 1. - beta2)
                elif p.dim() == 2 and not is_embedding:
                    # Sinkhorn Path for 2D Tensors
                    sinkhorn_lr = lr * self.sinkhorn_scale
                    if wd != 0:
                        p.add_(p, alpha=-sinkhorn_lr * wd)
                    update = grad.clone()
                    rows, cols = update.shape
                    for _ in range(L):
                        row_norm = torch.linalg.norm(update, ord=2, dim=1, keepdim=True)
                        update = math.sqrt(cols) * update / row_norm.add_(eps)
                        col_norm = torch.linalg.norm(update, ord=2, dim=0, keepdim=True)
                        update = math.sqrt(rows) * update / col_norm.add_(eps)
                    p.add_(update, alpha=-sinkhorn_lr)

                else:
                    # AdamW Path for 1D Tensors
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    state['step'] += 1

                    if wd != 0:
                        p.add_(p, alpha=-lr * wd)

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias1, bias2 = 1.0 - beta1 ** state['step'], 1.0 - beta2 ** state['step']
                    step_size = lr * math.sqrt(bias2) / bias1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
