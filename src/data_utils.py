# src/data_utils.py
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
from .config import DataArgs


def create_tokenized_dataset(data_args: DataArgs):
    """Loads a raw dataset, tokenizes and groups it."""
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    raw_datasets = load_from_disk(data_args.dataset_path)
    sampled_datasets = DatasetDict({
        split: ds.train_test_split(test_size=1.0-data_args.portion, seed=42)["train"]  # keep only 10%
        for split, ds in raw_datasets.items()
    })

    def tokenize_function(examples):
        return tokenizer(examples["contents"]) if 'contents' in examples else tokenizer(examples["text"])

    tokenized_datasets = sampled_datasets.map(
        tokenize_function, batched=True, remove_columns=sampled_datasets["train"].column_names
    )
    
    def group_texts(examples):
        block_size = data_args.block_size
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"][:]
        return result

    return tokenized_datasets.map(group_texts, batched=True)
