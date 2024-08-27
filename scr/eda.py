from typing import List, Any
import pandas as pd
from pandas import DataFrame
from matplotlib.figure import Figure
from datasets import load_dataset

def download_data(dataset_name: str) -> pd.DataFrame:
    """
    Download a dataset from Hugging Face and convert it to a pandas DataFrame.

    Parameters:
    dataset_name (str): The name of the dataset to download.

    Returns:
    pd.DataFrame: The downloaded dataset as a pandas DataFrame.
    """
    dataset = load_dataset(dataset_name)
    df = dataset['train'].to_pandas()
    return df


def generate_figures(df: DataFrame, review_column: str = 'text', img_dir: str = 'img') -> List[Figure]:
    """
    Generate and save figures based on the reviews in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the reviews.
    review_column (str): The name of the column in df that contains the reviews. Default is 'review'.
    img_dir (str): The directory where the figures will be saved. Default is 'img'.

    Returns:
    List[Figure]: A list of the generated figures.
    """
    import os
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # Ensure the img_dir exists
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Plot a histogram of review lengths
    fig1 = plt.figure()
    df[review_column].str.len().hist(edgecolor='black')
    plt.title('Histogram of Review Lengths')
    plt.xlabel('Review Length')
    plt.ylabel('Count')
    plt.savefig(os.path.join(img_dir, 'review_lengths.png'))
    plt.close(fig1)

    # Generate a word cloud of reviews
    fig2 = plt.figure(figsize=(10, 5))
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(df[review_column]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(img_dir, 'wordcloud.png'))
    plt.close(fig2)

    return [fig1, fig2]

if __name__ == "__main__":
    # Your code here
    df = download_data("theArijitDas/Fake-Reviews-Dataset")
    generate_figures(df)