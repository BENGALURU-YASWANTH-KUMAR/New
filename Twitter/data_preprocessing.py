import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")


def preprocess_text(df):
    port_stem = PorterStemmer()

    def stemming(content):
        stemmed_content = re.sub("[^a-zA-Z]", " ", content)
        stemmed_content = stemmed_content.lower().split()
        stemmed_content = [
            port_stem.stem(word)
            for word in stemmed_content
            if word not in stopwords.words("english")
        ]
        stemmed_content = " ".join(stemmed_content)
        return stemmed_content

    df["stemmed_content"] = df["text"].apply(stemming)
    return df


if __name__ == "__main__":
    df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1")
    df = preprocess_text(df)
    df.to_csv("preprocessed_tweets.csv", index=False)
    print(df.head())
