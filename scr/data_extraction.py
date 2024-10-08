from datasets import load_dataset

def load_hf_dataset(dataset_name='yelp_review_full': str):
    # Load Yelp Review Dataset
    dataset_name = load_dataset(dataset_name)

    return dataset