import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Twitter.feature_extraction import extract_features


def build_and_evaluate_model(df):
    X_train_vectorized, X_test_vectorized, y_train, y_test = extract_features(df)

    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_tweets.csv")
    build_and_evaluate_model(df)
