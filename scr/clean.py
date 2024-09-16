import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.corpus import stopwords
import joblib
from typing import Tuble

from eda import download_data

# Download NLTK data
nltk.download('stopwords')

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the data by cleaning the text, converting it to numerical features using TF-IDF,
    and splitting it into training and testing sets.

    Args:
        df (pd.DataFrame): The input dataframe containing the 'review' and 'label' columns.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the preprocessed feature matrix (X) and the labels (y).
    """
    # Function to clean text
    def clean_text(text: str) -> str:
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text

    # Apply the cleaning function to the review column
    df['cleaned_review'] = df['review'].apply(clean_text)

    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review']).toarray()

    # Labels
    y = df['label'].values

    return X, y

# Import the dataframe


if __name__ == "__main__":

    df = download_data("theArijitDas/Fake-Reviews-Dataset")

    # Preprocess the data
    X, y = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessed data
    joblib.dump((X_train, X_test, y_train, y_test), 'preprocessed_data.joblib')