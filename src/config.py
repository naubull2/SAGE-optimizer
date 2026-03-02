# src/config.py
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelArgs:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model."})

@dataclass
class DataArgs:
    dataset_path: str = field(metadata={"help": "Path to the raw or processed dataset."})
    tokenizer_name: str = field(default=None, metadata={"help": "Tokenizer to use. Defaults to model path."})
    block_size: int = field(default=512, metadata={"help": "Context size."})
    processed_output_path: str = field(default="./processed_dataset", metadata={"help": "Preprocessed data save path."})
    portion: float = field(default=1.0, metadata={"help": "Portion of the data if selection is needed."})

@dataclass
class ScriptArgs:
    """Custom arguments for our script that are not part of TrainingArguments."""
    optimizer: Literal["AdamW", "SinkGD_pure", "SinkGD", "APOLLO", "Lion", "SAGE_pure", "SAGE", "SAGE_lion", "SAGE_hybrid"] = field(default="SAGE", metadata={"help": "The optimizer to use."})
    is_pretrain: bool = field(default=True, metadata={"help": "Whether to treat the run as a pretrain from scratch. If set to True, the model weights will be reinitialized."})
    betas: str = field(default="0.9,0.99", metadata={"help": "Betas to use. Separated with comma. e.g., 0.9,0.99"})
    sinkhorn_scale: float = field(default=1, metadata={"help": "Sinkhorn scale alpha to tune SinkGD learning rate"})
    adaptive_sign: bool = field(default=True, metadata={"help": "Whether to use the adaptive sign technique or resort to plain Lion update"})
    hybrid: bool = field(default=True, metadata={"help": "Whether to use the Hybrid structure of using Embedding targeted optimizer for SAGE"})
    damper: Literal["sqrt", "log", None] = field(default="sqrt", metadata={"help": "The choice of damper function."})
