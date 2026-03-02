import locale
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch.optim as optim
import transformers
from datasets import load_from_disk
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                            DataCollatorForLanguageModeling, HfArgumentParser,
                            Trainer, TrainingArguments)
from transformers.integrations import TensorBoardCallback

import src.optimizers as custom_optimizers
from src.callbacks import MemoryUsageCallback, PerplexityCallback, RawVectorStateCallback
from src.config import DataArgs, ModelArgs, ScriptArgs

logger = logging.getLogger(__name__)


def format_number(num: int) -> str:
    """
    Convert a large number into a human-readable format
    (e.g., 1.2K, 100M)
    """
    units = [
        ("T", 1_000_000_000_000),
        ("B", 1_000_000_000),
        ("M", 1_000_000),
        ("K", 1_000)
    ]
    for suffix, factor in units:
        if num >= factor:
            val = num / factor
            return f"{val:.0f}{suffix}" if val.is_integer() else f"{val:.1f}{suffix}"

    return str(num)  # smaller than 1K → raw number


def analyze_model_parameters(model):
    """
    Loads a model and prints a summary of its parameter distribution.
    """
    print(f"Analyzing model parameters\n")

    param_counts = defaultdict(int)
    total_params = 0

    # Group parameters by their main block
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "embed_tokens" in name or "wte" in name or "wpe" in name:
            param_counts["Embedding"] += param.numel()
        elif "lm_head" in name or "output" in name:
            param_counts["Output Projection (LM Head)"] += param.numel()
        elif ".layers." in name or ".h." in name or ".block." in name:
            layer_num = name.split('.')[2]
            param_counts[f"Transformer Block {layer_num}"] += param.numel()
        else:
            param_counts["Other"] += param.numel()

    print(f"Total Model Parameters: {format_number(total_params)} ({locale.format_string('%d', total_params, grouping=True)})\n")

    print("Parameter Distribution Summary")
    # Sort for readability, putting major groups first
    sorted_items = sorted(
        param_counts.items(),
        key=lambda item: (
            "Embedding" not in item[0],
            "Output Projection (LM Head)" not in item[0],
            item[0]
        )
    )
    for name, count in sorted_items:
        percentage = (count / total_params) * 100
        print(f"{name:<35}: {locale.format_string('%12d', count, grouping=True)} params ({percentage:5.2f}%)")


def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, ScriptArgs, TrainingArguments))
    model_args, data_args, script_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer_path = data_args.tokenizer_name or model_args.model_name_or_path
    config_path = model_args.model_name_or_path
    
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # we need pad token for batch packing 

    logger.info(f"Loading pre-processed dataset from: {data_args.dataset_path}")
    lm_datasets = load_from_disk(data_args.dataset_path)

    if script_args.is_pretrain:
        logger.info(f"Initializing model from config: {config_path}")
        config = AutoConfig.from_pretrained(config_path)
        model = getattr(transformers, config.architectures[0])(config)
    else:
        logger.info(f"Loading pretrained model from: {model_args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    analyze_model_parameters(model)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create the custom optimizer
    optimizer = None
    betas = tuple(float(v) for v in script_args.betas.split(','))
    
    if script_args.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=training_args.learning_rate, 
            weight_decay=training_args.weight_decay,
            betas=betas
        )

    elif hasattr(custom_optimizers, script_args.optimizer):
        optimizer_class = getattr(custom_optimizers, script_args.optimizer)
        if script_args.optimizer == "APOLLO":
            # APOLLO has special param group requirements
            lowrank_params = []
            non_lowrank_params = []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    if p.dim() == 2 and ('.attn.' in n or '.mlp.' in n):
                        lowrank_params.append(p)
                    else:
                        non_lowrank_params.append(p)
            param_groups = [
                {"params": non_lowrank_params},
                {"params": lowrank_params,
                 "rank": 1, "proj": "random", "scale_type": "tensor",
                 "scale": 128, "update_proj_gap": 200, "proj_type": "std"
                }
            ]
            optimizer = optimizer_class(
                param_groups, 
                lr=training_args.learning_rate
            )
            
        elif script_args.optimizer == "Lion":
            optimizer = optimizer_class(
                model.parameters(),
                lr=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
                betas=betas
            )
            
        else:
            # Default path for SAGE_global, SAGE_hybrid, SAGE_pure, SinkGD_pure, SinkGD
            named_param_groups = [{'params': [p], 'name': n}
                                  for n, p in model.named_parameters() if p.requires_grad]
            optimizer = optimizer_class(
                named_param_groups,
                lr=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
                betas=betas,
                sinkhorn_scale=script_args.sinkhorn_scale,
            )
    else:
        raise ValueError(f"Optimizer '{script_args.optimizer}' not found in torch.optim or custom_optimizers.")


    memory_callback = MemoryUsageCallback()
    ppl_callback = PerplexityCallback()
    tensorboard_callback = TensorBoardCallback()
    #s_range_callback = RawVectorStateCallback(tensorboard_callback,
    #                                          save_dir="./raw_logs", target_param_name="s_range")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"].shuffle(seed=training_args.seed)[:1024],
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[memory_callback, ppl_callback, tensorboard_callback]
        #callbacks=[memory_callback, ppl_callback, s_range_callback, tensorboard_callback]
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
