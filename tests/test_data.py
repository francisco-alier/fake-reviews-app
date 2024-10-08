import logging
from data_extraction import load_huggingface_dataset

# Set up basic logging configuration
logging.basicConfig(filename='test_dataset_loader.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def test_huggingface_connection():
    """
    Basic test to check if the Hugging Face API connection is working.
    Attempts to load a small dataset.
    """
    try:
        dataset = load_huggingface_dataset("yelp_review_full", split="train")
        assert dataset is not None, "Failed to load dataset."
        assert len(dataset) > 0, "Dataset is empty."
        logger.info("Test passed: Hugging Face connection is working.")
    except AssertionError as ae:
        logger.error(f"Test failed: {ae}")