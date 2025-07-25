from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
    flash,
    jsonify,
)
import pickle
import numpy as np
import pandas as pd
from textblob import TextBlob
import pyrebase
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from dotenv import load_dotenv
import os
import requests


load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_secret_key")

# Configure Google Generative AI
import requests
import json

GEMINI_API_KEY = os.getenv("gemini_api_key")
if not GEMINI_API_KEY:
    print("Warning: Gemini API key not found in environment variables")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def generate_gemini_response(prompt):
    headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}

    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return None


# Firebase Configuration (load from environment variables)
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL", ""),
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Load models and data
popular_df = pickle.load(open("BookRecommender/popular.pkl", "rb"))
pt = pickle.load(open("BookRecommender/pt.pkl", "rb"))
books = pickle.load(open("BookRecommender/books.pkl", "rb"))
similarity_scores = pickle.load(open("BookRecommender/similarity_scores.pkl", "rb"))

ps = PorterStemmer()

# Fix sentiment_model loading to ensure it is a model object, not a numpy array
try:
    import joblib

    sentiment_model = joblib.load("Twitter/sentiment_rf_model.joblib")
except Exception:
    sentiment_model = pickle.load(open("Twitter/sentiment_rf_model.pkl", "rb"))


def stemming(content):
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        ps.stem(word)
        for word in stemmed_content
        if word not in stopwords.words("english")
    ]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content


@app.route("/")
def index():
    return render_template(
        "index.html",
        book_name=list(popular_df["Book-Title"].values),
        author=list(popular_df["Book-Author"].values),
        image=list(popular_df["Image-URL-M"].values),
        votes=list(popular_df["num_rating"].values),
        rating=list(popular_df["avg_rating"].values),
    )


@app.route("/recommend")
def recommend_ui():
    return render_template("recommend.html")


@app.route("/recommend_books", methods=["post"])
def recommend():
    user_input = request.form.get("user_input")
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True
    )[1:10]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books["Book-Title"] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Title"].values))
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Author"].values))
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Image-URL-M"].values))
        data.append(item)

    return render_template("recommend.html", data=data)


@app.route("/popular")
def popular():
    return render_template(
        "popular.html",
        book_name=list(popular_df["Book-Title"].values),
        author=list(popular_df["Book-Author"].values),
        image=list(popular_df["Image-URL-M"].values),
        votes=list(popular_df["num_rating"].values),
        rating=list(popular_df["avg_rating"].values),
        books=books,
    )


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/twitter_sentiment")
def twitter_sentiment():
    return render_template("twitter_sentiment.html")


@app.route("/predict_sentiment", methods=["post"])
def predict_sentiment():
    tweet_text = request.form.get("tweet_text")
    processed_tweet = stemming(tweet_text)
    # Use TextBlob to get sentiment_score
    sentiment_score = TextBlob(tweet_text).sentiment.polarity
    input_df = pd.DataFrame(
        {"stemmed_content": [processed_tweet], "sentiment_score": [sentiment_score]}
    )
    try:
        prediction = sentiment_model.predict(input_df)[0]
        if prediction == 1:
            sentiment = "Positive"
        elif prediction == -1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
    except Exception as e:
        sentiment = f"Error: {str(e)}"
    return render_template(
        "twitter_sentiment.html", sentiment=sentiment, tweet_text=tweet_text
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session["user"] = user["idToken"]
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        except Exception:
            flash("Invalid email or password", "danger")
            return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            auth.create_user_with_email_and_password(email, password)
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("login"))
        except Exception:
            flash("Email already exists.", "danger")
            return redirect(url_for("signup"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        if "user" not in session:
            flash("Please login to use the chatbot.", "warning")
            return redirect(url_for("login"))
        return render_template("chatbot.html")

    if request.method == "POST":
        try:
            if "user" not in session:
                return jsonify({"error": "Please login first"}), 401

            message = request.json.get("message")
            if not message:
                return jsonify({"error": "No message provided"}), 400

            # Generate response from Gemini
            api_response = generate_gemini_response(message)

            # Check if response generation was successful
            if api_response and "candidates" in api_response:
                response_text = api_response["candidates"][0]["content"]["parts"][0][
                    "text"
                ]
                return jsonify({"response": response_text, "status": "success"})
            else:
                return jsonify(
                    {"error": "Could not generate response. Please try again."}
                ), 500

        except Exception as e:
            print(f"Chat error: {str(e)}")  # Log the error
            return jsonify(
                {
                    "error": "An error occurred while processing your request. Please try again."
                }
            ), 500


@app.route("/chat_suggestions")
def chat_suggestions():
    suggestions = [
        "📚 Recommend me a good book to read",
        "🔍 What's the plot of 1984 by George Orwell?",
        "💻 Help me learn Python programming",
        "🧮 Explain quantum physics simply",
        "📖 Tell me about world history",
        "🍳 Share a pasta recipe",
        "🌍 Facts about climate change",
        "🎵 History of jazz music",
        "💡 Creative writing tips",
        "🎨 Famous art movements",
    ]
    return jsonify(suggestions)


if __name__ == "__main__":
    app.run(debug=True)
