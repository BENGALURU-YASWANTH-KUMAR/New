import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def extract_features(df):
    X = df["stemmed_content"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1111, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return X_train_vectorized, X_test_vectorized, y_train, y_test


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_tweets.csv")
    X_train_vectorized, X_test_vectorized, y_train, y_test = extract_features(df)
    print(X_train_vectorized.shape, X_test_vectorized.shape)
