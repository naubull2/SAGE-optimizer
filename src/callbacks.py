import logging
import math
import os
from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    Trainer
)
import torch


logger = logging.getLogger(__name__)


def measure_memory_usage(model: torch.nn.Module, optimizer:torch.optim.Optimizer) -> dict:
    """Measures memory usage of the model and optimizer."""
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    
    mem_optimizer = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                param_state = optimizer.state[p]
                for state_val in param_state.values():
                    if isinstance(state_val, torch.Tensor):
                        mem_optimizer += state_val.nelement() * state_val.element_size()

    mem_total_gb = (mem_params + mem_bufs + mem_optimizer) / (1024**3)
    
    return {
        "model_params_gb": mem_params / (1024**3),
        "optimizer_states_gb": mem_optimizer / (1024**3),
        "total_estimated_gb": mem_total_gb
    }


class MemoryUsageCallback(TrainerCallback):
    """
    A callback to measure and log memory usage. It measures memory after the first
    step and injects metrics into the log.
    """
    _memory_stats = {}

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        if model and optimizer:
            self._memory_stats = measure_memory_usage(model, optimizer)

        if state.global_step == 1:
            logger.info(f"Memory Usage: {self._memory_stats}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict, **kwargs):
        if self._memory_stats:
            logs.update(self._memory_stats)


class PerplexityCallback(TrainerCallback):
    """
    Computes and logs perplexity at the end of each evaluation.
    """
    def __init__(self):
        self._eval_loss = None

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs):
        for key in metrics.keys():
            if "eval_loss" in key or 'labels_loss' in key:
                self._eval_loss = metrics[key]
                break

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict, **kwargs):
        """Inject perplexity into the logs right before they are written."""
        if self._eval_loss is not None:
            try:
                perplexity = math.exp(self._eval_loss)
                logs["eval_ppl"] = perplexity
            except OverflowError:
                logs["eval_ppl"] = float("inf")
            self._eval_loss = None


class DirectionalSharpnessCallback(TrainerCallback):
    """
    Logs directional sharpness during training using finite difference approximation.
    """
    def __init__(self, sharpness_epsilon=1e-3, log_steps=100):
        self.sharpness_epsilon = sharpness_epsilon
        self.log_steps = log_steps

    def on_step_end(self, args, state, control, model, tokenizer, optimizer, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_steps == 0:
            if not isinstance(model.module, torch.nn.Module):
                model = model.module
            
            # Capture optimization direction using normalized gradient proxy
            
            original_weights = [p.data.clone() for p in model.parameters()]
            
            # Calculate loss at current weights (L(w))
            inputs = kwargs['inputs']
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                loss_w = outputs.loss
            
            # Get current gradients
            if any(p.grad is None for p in model.parameters()):
                model(**inputs)[0].backward()
            
            grad_list = [p.grad.data.clone() for p in model.parameters() if p.grad is not None]
            grad_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in grad_list]))
            
            if grad_norm == 0:
                logger.info(f"Gradient norm is zero at step {state.global_step}, skipping sharpness calculation.")
                control.log_history.append({"directional_sharpness": 0.0, "step": state.global_step})
                return
                
            # Create unit direction vector
            direction_vector = [g / grad_norm for g in grad_list]

            # Calculate loss at perturbed weights (L(w + eps*v))
            for p, d in zip(model.parameters(), direction_vector):
                p.data.add_(d, alpha=self.sharpness_epsilon)
            
            with torch.no_grad():
                outputs_plus = model(**inputs)
                loss_w_plus = outputs_plus.loss
                
            # Calculate loss at perturbed weights (L(w - eps*v))
            for p, d in zip(model.parameters(), direction_vector):
                p.data.add_(d, alpha=-2*self.sharpness_epsilon)
            
            with torch.no_grad():
                outputs_minus = model(**inputs)
                loss_w_minus = outputs_minus.loss
            
            # Restore original weights
            for p, original_p in zip(model.parameters(), original_weights):
                p.data.copy_(original_p)

            # Calculate directional sharpness
            directional_sharpness = (loss_w_plus - 2 * loss_w + loss_w_minus) / (self.sharpness_epsilon ** 2)
            
            logs = {"directional_sharpness": directional_sharpness.item()}
            control.log_history.append(logs)
            logger.info(f"Step {state.global_step}: Directional Sharpness = {directional_sharpness.item():.4f}")


class RawVectorStateCallback(TrainerCallback):
    """
    Logs to TensorBoard and saves raw tensors for analysis.
    """
    def __init__(self, tb_callback, save_dir="./raw_logs", target_param_name="s_range"):
        super().__init__()
        self.save_dir = save_dir
        self.target_param_name = target_param_name
        os.makedirs(self.save_dir, exist_ok=True)
        self.tb_callback = tb_callback
        self.tb_writer = None
        self.h_t_history = []

    def _get_writer(self):
        self.tb_writer = self.tb_callback.tb_writer

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.tb_writer is None:
            self._get_writer()

        if state.global_step % args.logging_steps == 0:
            optimizer = kwargs.get("optimizer")
            if optimizer is None: return

            # Access H_t vector
            h_t_vector = None
            for group in optimizer.param_groups:
                if 'embed' in group.get("name").lower():
                    first_param = group['params'][0]
                    if first_param in optimizer.state:
                        h_t_vector = optimizer.state[first_param].get(self.target_param_name)
            
            if h_t_vector is not None:
                ## --- 1. Live Monitoring to TensorBoard ---
                ## Log summary stats
                #self.tb_writer.add_scalar(f"{self.target_param_name}/mean", h_t_vector.mean(), state.global_step)
                ## Log the histogram for interactive viewing
                #self.tb_writer.add_histogram(f"{self.target_param_name}/distribution", h_t_vector, state.global_step)

                # --- 2. Save Raw Data for Matplotlib ---
                # Save the raw vector for creating a high-fidelity histogram later
                save_path = os.path.join(self.save_dir, f"{self.target_param_name}_step_{state.global_step}.pt")
                torch.save(h_t_vector.cpu(), save_path)
                
                # Append to list for final heatmap
                self.h_t_history.append(h_t_vector.cpu().clone())


    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Save full history
        if self.h_t_history:
            heatmap_data = torch.stack(self.h_t_history) # Shape: [num_steps, d]
            save_path = os.path.join(self.save_dir, f"{self.target_param_name}_heatmap_data.pt")
            torch.save(heatmap_data, save_path)
            logger.info(f"Saved heatmap data to {save_path}")
