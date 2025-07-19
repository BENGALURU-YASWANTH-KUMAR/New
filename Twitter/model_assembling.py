import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib


def build_final_model(df):
    X = df[["stemmed_content", "sentiment_score"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000), "stemmed_content"),
            ("num", StandardScaler(), ["sentiment_score"]),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "final_model.joblib")

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Model Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_tweets.csv")
    build_final_model(df)
