import os
import logging
from datasets import load_dataset, concatenate_datasets, DatasetDict


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Define the list of datasets to process
# NOTE: this is a sample subset for sanity check, one needs much larger dataset for proper LLM pretraining.
DATASET_NAMES = [
    "timaeus/pile_subsets_mini"
]

LOCAL_PATH = "./mini_pile_subset"

# Define split ratios and a random seed
TRAIN_SIZE = 0.8
SHUFFLE_SEED = 42


def main():
    all_train_splits, all_validation_splits, all_test_splits = [], [], []

    logger.info("Starting Dataset Processing")

    # Load, Split, and Stratify each dataset
    for name in DATASET_NAMES:
        try:
            logger.info(f"\n- Processing '{name}'...")
            
            # Load the original 'train' split
            dataset = load_dataset(name, split='train')

            # Split off the largest chunk
            train_valtest_split = dataset.train_test_split(
                train_size=TRAIN_SIZE, shuffle=True, seed=SHUFFLE_SEED
            )
            all_train_splits.append(train_valtest_split['train'])

            # Split the remaining pool into validation and test
            val_test_pool = train_valtest_split['test']
            val_test_split = val_test_pool.train_test_split(
                test_size=0.5, shuffle=True, seed=SHUFFLE_SEED
            )

            all_validation_splits.append(val_test_split['train'])
            all_test_splits.append(val_test_split['test'])
            
            logger.info(f"Finished splitting '{name}'.")

        except Exception as e:
            logger.error(f"Failed to process '{name}'. Reason: {e}")
            continue

    # --- Step 3: Concatenate all splits into global datasets ---
    if not all_train_splits:
        logger.warning("\nNo datasets were processed. Exiting.")
        return

    logger.info("Combining all stratified splits...")
    
    final_train_set = concatenate_datasets(all_train_splits)
    final_validation_set = concatenate_datasets(all_validation_splits)
    final_test_set = concatenate_datasets(all_test_splits)

    # Create a DatasetDict and save to disk
    final_dataset_dict = DatasetDict({
        'train': final_train_set,
        'validation': final_validation_set,
        'test': final_test_set
    })
    
    logger.info(f"Final Dataset Splits:")
    logger.info(final_dataset_dict)
    
    logger.info(f"Saving final DatasetDict to '{LOCAL_PATH}'...")
    final_dataset_dict.save_to_disk(LOCAL_PATH)
    logger.info("Successfully saved.")

    # Clean up the original downloaded cache
    logger.info("Starting cache cleanup...")
    total_reclaimed_gb = 0
    for name in DATASET_NAMES:
        try:
            ds_to_clean = load_dataset(name, split='train')
            cleanup_info = ds_to_clean.cleanup_cache_files()
            
            # FIX for the AttributeError
            if isinstance(cleanup_info, dict):
                reclaimed_size_bytes = cleanup_info.get('reclaimed_size', 0)
                num_files = cleanup_info.get('num_files', 0)
            else: # It's an integer
                reclaimed_size_bytes = cleanup_info
                num_files = "N/A" # File count is not available in this case
            
            reclaimed_size_gb = reclaimed_size_bytes / (1024**3)
            total_reclaimed_gb += reclaimed_size_gb
            
            logger.info(
                f"  - Cleaned '{name}': Removed {num_files} files, reclaiming {reclaimed_size_gb:.2f} GB."
            )
        except Exception as e:
            logger.error(f"  - Could not clean up '{name}'. Reason: {e}")

    logger.info(f"Total space reclaimed: {total_reclaimed_gb:.2f} GB")
    logger.info("All Done!")


if __name__ == "__main__":
    main()
