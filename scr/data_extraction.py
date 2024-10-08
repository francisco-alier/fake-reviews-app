import logging
from datasets import load_dataset

# Set up basic logging configuration
logging.basicConfig(filename='dataset_loader.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def load_huggingface_dataset(dataset_name: str = "yelp_review_full", split: str = None, cache_dir: str = None):
    """
    Loads a dataset from Hugging Face based on the provided dataset name.
    
    Parameters:
        dataset_name (str): The name of the dataset to load (e.g., "yelp_review_full").
        split (str): The data split to load (e.g., "train", "test"). Optional.
        cache_dir (str): Directory to cache the dataset. Optional.
    
    Returns:
        dataset: Loaded Hugging Face dataset.
    """
    try:
        if split:
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        logger.info(f"Successfully loaded dataset: {dataset_name}")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return None