import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from Twitter.feature_extraction import extract_features


def optimize_model(df):
    X_train_vectorized, X_test_vectorized, y_train, y_test = extract_features(df)

    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer(max_features=5000)), ("clf", LogisticRegression())]
    )

    param_dist = {"tfidf__max_features": [5000, 10000, 20000], "clf__C": [0.1, 1, 10]}

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        random_state=42,
    )
    random_search.fit(X_train_vectorized, y_train)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Optimized Model Accuracy: {accuracy:.4f}")
    print(f"Best Parameters: {random_search.best_params_}")


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_tweets.csv")
    optimize_model(df)
