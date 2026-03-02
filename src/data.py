from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokenized_dataset(data_args: 'DataArgs'):
    """Loads dataset, tokenizes it, and prepares it for training."""
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    
    raw_datasets = load_dataset(data_args.dataset_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    
    block_size = data_args.block_size

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the small remainder
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"][:]
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    return lm_datasets, tokenizer
