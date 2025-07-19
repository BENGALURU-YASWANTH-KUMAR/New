import pandas as pd
import matplotlib.pyplot as plt


def visualize_data(df):
    df["target"].value_counts().plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_tweets.csv")
    visualize_data(df)
