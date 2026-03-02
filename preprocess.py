import argparse
import logging
from pathlib import Path
from transformers import HfArgumentParser
from src.config import DataArgs
from src.data_utils import create_tokenized_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Runs data tokenization and processing.
    """
    parser = HfArgumentParser((DataArgs,))
    data_args, = parser.parse_args_into_dataclasses()

    logger.info(f"Starting preprocessing for dataset at: {data_args.dataset_path}")

    lm_datasets = create_tokenized_dataset(data_args)

    processed_dataset_path = data_args.processed_output_path
    
    logger.info(f"Saving tokenized and processed dataset to: {processed_dataset_path}")
    
    lm_datasets.save_to_disk(processed_dataset_path)
    
    logger.info("Preprocessing complete!")
    logger.info(f"You can now point your training script to '{processed_dataset_path}'")


if __name__ == "__main__":
    main()
