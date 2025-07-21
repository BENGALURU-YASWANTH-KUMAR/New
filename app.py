from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
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
import nltk

# NEW IMPORTS FOR ENHANCED CHATBOT
import google.generativeai as genai
import json
import threading
import time
from datetime import datetime

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback_secret_key_for_development')

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

# Initialize Firebase only if config is available
firebase = None
auth = None
try:
    if all(firebase_config.values()):
        firebase = pyrebase.initialize_app(firebase_config)
        auth = firebase.auth()
        print("✅ Firebase initialized successfully!")
    else:
        print("⚠️ Firebase config incomplete - auth features disabled")
except Exception as e:
    print(f"⚠️ Firebase initialization failed: {e}")

# Load models and data with error handling
try:
    popular_df = pickle.load(open("BookRecommender/popular.pkl", "rb"))
    pt = pickle.load(open("BookRecommender/pt.pkl", "rb"))
    books = pickle.load(open("BookRecommender/books.pkl", "rb"))
    similarity_scores = pickle.load(open("BookRecommender/similarity_scores.pkl", "rb"))
    print("✅ Book recommendation models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading book models: {e}")
    # Create dummy data to prevent crashes
    popular_df = pd.DataFrame()
    pt = pd.DataFrame()
    books = pd.DataFrame()
    similarity_scores = np.array([])

ps = PorterStemmer()

# Fix sentiment_model loading to ensure it is a model object, not a numpy array
sentiment_model = None
try:
    import joblib
    sentiment_model = joblib.load("Twitter/sentiment_rf_model.joblib")
    print("✅ Sentiment model loaded successfully!")
except Exception:
    try:
        sentiment_model = pickle.load(open("Twitter/sentiment_rf_model.pkl", "rb"))
        print("✅ Sentiment model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading sentiment model: {e}")

# ENHANCED UNIVERSAL CHATBOT CLASS
class UniversalChatbot:
    def __init__(self, books_data, popular_df):
        self.books_data = books_data
        self.popular_df = popular_df
        self.setup_gemini()
        self.conversation_history = []
        
    def setup_gemini(self):
        """Initialize Gemini AI"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("Warning: GEMINI_API_KEY not found in environment variables")
                self.model = None
                return
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                "gemini-2.0-flash-exp",
                generation_config={
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "max_output_tokens": 1200,
                }
            )
            print("✅ Gemini AI initialized successfully!")
        except Exception as e:
            print(f"❌ Error setting up Gemini: {e}")
            self.model = None
    
    def detect_intent(self, message):
        """Enhanced intent detection for any type of question"""
        message_lower = message.lower()
        
        # Check for specific intents but don't limit to them
        if any(word in message_lower for word in ['book', 'read', 'author', 'recommend', 'novel', 'literature']):
            return 'book_related'
        elif any(word in message_lower for word in ['sentiment', 'twitter', 'analyze', 'emotion', 'feeling']):
            return 'sentiment_analysis'
        elif any(word in message_lower for word in ['popular', 'trending', 'top books', 'bestseller']):
            return 'popular_books'
        elif any(word in message_lower for word in ['math', 'calculate', 'solve', 'equation', 'number']):
            return 'mathematics'
        elif any(word in message_lower for word in ['science', 'physics', 'chemistry', 'biology', 'technology']):
            return 'science_tech'
        elif any(word in message_lower for word in ['history', 'past', 'ancient', 'war', 'historical']):
            return 'history'
        elif any(word in message_lower for word in ['programming', 'code', 'python', 'javascript', 'software']):
            return 'programming'
        elif any(word in message_lower for word in ['health', 'medical', 'doctor', 'symptoms', 'medicine']):
            return 'health'
        elif any(word in message_lower for word in ['recipe', 'cook', 'food', 'ingredient', 'meal']):
            return 'cooking'
        elif any(word in message_lower for word in ['travel', 'place', 'country', 'city', 'visit']):
            return 'travel'
        elif any(word in message_lower for word in ['weather', 'climate', 'temperature', 'rain', 'sunny']):
            return 'weather'
        else:
            return 'general_knowledge'
    
    def get_book_recommendations(self, query):
        """Get book recommendations based on query"""
        try:
            if self.books_data.empty:
                return []
            # Search for books in your dataset
            matching_books = self.books_data[
                self.books_data['Book-Title'].str.contains(query, case=False, na=False) |
                self.books_data['Book-Author'].str.contains(query, case=False, na=False)
            ].head(5)
            
            recommendations = []
            for _, book in matching_books.iterrows():
                recommendations.append({
                    'title': book['Book-Title'],
                    'author': book['Book-Author'],
                    'image': book.get('Image-URL-M', '')
                })
            
            return recommendations
        except Exception as e:
            print(f"Error getting book recommendations: {e}")
            return []
    
    def get_popular_books(self):
        """Get popular books from your dataset"""
        try:
            if self.popular_df.empty:
                return []
            popular_books = []
            for i in range(min(5, len(self.popular_df))):
                popular_books.append({
                    'title': self.popular_df.iloc[i]['Book-Title'],
                    'author': self.popular_df.iloc[i]['Book-Author'],
                    'rating': float(self.popular_df.iloc[i]['avg_rating']),
                    'votes': int(self.popular_df.iloc[i]['num_rating']),
                    'image': self.popular_df.iloc[i]['Image-URL-M']
                })
            return popular_books
        except Exception as e:
            print(f"Error getting popular books: {e}")
            return []
    
    def create_universal_prompt(self, message, intent, context_data=None):
        """Create context-aware prompt for any type of question"""
        base_context = f"""
        You are an intelligent AI assistant that can answer ANY question on ANY topic.
        You have access to a Book Recommendation platform with 271,360 books and Twitter Sentiment Analysis tools.
        Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        You should:
        1. Answer ANY question the user asks - don't limit yourself to books or sentiment analysis
        2. Be helpful, accurate, and comprehensive
        3. Use appropriate emojis to make responses engaging
        4. Provide practical, useful information
        5. If asked about books, you can access the book database
        6. If asked about sentiment analysis, explain how it works
        7. For any other topic (math, science, history, cooking, etc.), provide detailed helpful answers
        
        Special capabilities:
        - 📚 Book recommendations from database
        - 😊 Sentiment analysis explanations  
        - 🧮 Mathematics and calculations
        - 🔬 Science and technology
        - 📖 History and culture
        - 💻 Programming and coding
        - 🍳 Recipes and cooking
        - ✈️ Travel advice
        - 🏥 General health information
        - 🎓 Educational content
        - And much more!
        """
        
        # Add specific context based on intent
        if intent == 'book_related' and context_data and context_data.get('books'):
            book_list = [f"{book['title']} by {book['author']}" for book in context_data['books'][:3]]
            book_context = f"\nFound relevant books in database: {', '.join(book_list)}"
            base_context += book_context
            
        elif intent == 'popular_books' and context_data and context_data.get('popular_books'):
            popular_list = [f"{book['title']} (⭐{book['rating']:.1f})" for book in context_data['popular_books'][:3]]
            popular_context = f"\nCurrent popular books: {', '.join(popular_list)}"
            base_context += popular_context
        
        return f"""{base_context}
        
        User Question: {message}
        
        Please provide a comprehensive, helpful answer. Use emojis appropriately and keep the response engaging and informative.
        If it's about books, leverage the database information. For any other topic, provide your best knowledge and advice.
        """
    
    def get_response(self, message, user_session=None):
        """Get AI response for ANY type of question"""
        if not self.model:
            return {
                'response': "❌ Sorry, AI assistant is currently unavailable. Please check the API configuration and try again later.",
                'intent': 'error'
            }
        
        try:
            intent = self.detect_intent(message)
            context_data = {}
            
            # Add relevant context based on intent
            if intent == 'book_related':
                context_data['books'] = self.get_book_recommendations(message)
            elif intent == 'popular_books':
                context_data['popular_books'] = self.get_popular_books()
            
            # Create universal prompt
            prompt = self.create_universal_prompt(message, intent, context_data)
            
            # Get AI response
            response = self.model.generate_content(prompt)
            
            # Store in conversation history (optional)
            self.conversation_history.append({
                'user_message': message,
                'bot_response': response.text if response.text else "No response generated",
                'intent': intent,
                'timestamp': datetime.now()
            })
            
            return {
                'response': response.text if response.text else "Sorry, I couldn't process that request.",
                'intent': intent,
                'context_data': context_data
            }
            
        except Exception as e:
            print(f"Chatbot error: {e}")
            return {
                'response': f"❌ I encountered an error while processing your question. Please try asking again or rephrase your question. (Error: {str(e)[:50]}...)",
                'intent': 'error'
            }

# Initialize universal chatbot after loading models
try:
    chatbot = UniversalChatbot(books, popular_df)
    print("🤖 Universal AI Chatbot initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize chatbot: {e}")
    chatbot = None

def stemming(content):
    try:
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
    except Exception as e:
        print(f"Error in stemming: {e}")
        return content.lower()

# EXISTING ROUTES
@app.route("/")
def index():
    try:
        if not popular_df.empty:
            return render_template(
                "index.html",
                book_name=list(popular_df["Book-Title"].values),
                author=list(popular_df["Book-Author"].values),
                image=list(popular_df["Image-URL-M"].values),
                votes=list(popular_df["num_rating"].values),
                rating=list(popular_df["avg_rating"].values),
            )
        else:
            return render_template("index.html", book_name=[], author=[], image=[], votes=[], rating=[])
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template("index.html", book_name=[], author=[], image=[], votes=[], rating=[])

@app.route("/recommend")
def recommend_ui():
    return render_template("recommend.html")

@app.route("/recommend_books", methods=["post"])
def recommend():
    try:
        user_input = request.form.get("user_input")
        if pt.empty or books.empty:
            return render_template("recommend.html", data=[], error="Recommendation system not available")
            
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
    except Exception as e:
        print(f"Error in recommend route: {e}")
        return render_template("recommend.html", data=[], error=f"Error: {str(e)}")

@app.route("/popular")
def popular():
    try:
        if not popular_df.empty:
            return render_template(
                "popular.html",
                book_name=list(popular_df["Book-Title"].values),
                author=list(popular_df["Book-Author"].values),
                image=list(popular_df["Image-URL-M"].values),
                votes=list(popular_df["num_rating"].values),
                rating=list(popular_df["avg_rating"].values),
                books=books,
            )
        else:
            return render_template("popular.html", book_name=[], author=[], image=[], votes=[], rating=[], books=[])
    except Exception as e:
        print(f"Error in popular route: {e}")
        return render_template("popular.html", book_name=[], author=[], image=[], votes=[], rating=[], books=[])

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/twitter_sentiment")
def twitter_sentiment():
    return render_template("twitter_sentiment.html")

@app.route("/predict_sentiment", methods=["post"])
def predict_sentiment():
    try:
        tweet_text = request.form.get("tweet_text")
        if not tweet_text:
            return render_template("twitter_sentiment.html", sentiment="Please enter text", tweet_text="")
            
        processed_tweet = stemming(tweet_text)
        # Use TextBlob to get sentiment_score
        sentiment_score = TextBlob(tweet_text).sentiment.polarity
        
        if sentiment_model is not None:
            input_df = pd.DataFrame(
                {"stemmed_content": [processed_tweet], "sentiment_score": [sentiment_score]}
            )
            prediction = sentiment_model.predict(input_df)[0]
            if prediction == 1:
                sentiment = "Positive"
            elif prediction == -1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
        else:
            # Fallback to TextBlob if model not available
            if sentiment_score > 0.1:
                sentiment = "Positive"
            elif sentiment_score < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
    except Exception as e:
        sentiment = f"Error: {str(e)}"
        
    return render_template(
        "twitter_sentiment.html", sentiment=sentiment, tweet_text=tweet_text
    )

# ENHANCED UNIVERSAL CHATBOT ROUTES
@app.route("/chatbot")
def chatbot_ui():
    """Render universal chatbot page"""
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle ANY type of chat message"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'response': '💭 Please ask me anything! I can help with books, science, math, cooking, history, or any other topic!',
                'intent': 'empty'
            })
        
        if not chatbot:
            return jsonify({
                'response': '❌ AI assistant is currently unavailable. Please check the configuration.',
                'intent': 'error'
            })
        
        # Get chatbot response for ANY question
        result = chatbot.get_response(message, session)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'response': f'❌ Error processing your question: {str(e)}. Please try again!',
            'intent': 'error'
        })

@app.route("/chat_suggestions")
def chat_suggestions():
    """Get diverse chat suggestions covering many topics"""
    suggestions = [
        "📚 Recommend books like Harry Potter",
        "🧮 Solve this math problem: 25 x 34",
        "🔬 Explain how photosynthesis works",
        "🍳 Recipe for chocolate cake",
        "📖 Tell me about World War 2",
        "💻 How to learn Python programming",
        "🏥 What are symptoms of common cold",
        "✈️ Best places to visit in Europe",
        "🌤️ Explain climate change",
        "😊 Analyze sentiment of text",
        "🎵 Who composed Beethoven's 9th Symphony",
        "🏃 Tips for running a marathon",
        "🌍 Facts about the solar system",
        "💡 How do electric cars work",
        "🎨 Famous Renaissance artists",
        "📱 Latest technology trends"
    ]
    return jsonify(suggestions)

@app.route("/chat_history")
def chat_history():
    """Get recent chat history"""
    try:
        if chatbot and hasattr(chatbot, 'conversation_history'):
            recent_history = chatbot.conversation_history[-10:]  # Last 10 conversations
            return jsonify(recent_history)
        return jsonify([])
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return jsonify([])

# AUTH ROUTES
@app.route("/login", methods=["GET", "POST"])
def login():
    if not auth:
        flash("Authentication service unavailable", "warning")
        return render_template("login.html", is_signup_page=False)
        
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session["user"] = user["idToken"]
            return jsonify({"success": True, "message": "Login successful"})
        except Exception as e:
            return jsonify({"success": False, "message": f"Invalid email or password."}), 401
    return render_template("login.html", is_signup_page=False)

@app.route("/google-login", methods=["POST"])
def google_login():
    if not auth:
        return jsonify({"error": "Authentication service unavailable"}), 503
        
    try:
        # Get the ID token from the request
        id_token = request.json.get("idToken")
        if not id_token:
            return jsonify({"error": "No ID token provided"}), 400
            
        # Verify the ID token with Firebase
        decoded_token = auth.verify_id_token(id_token)
        
        # Get user info
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        
        # Store user info in session
        session["user"] = id_token
        session["user_email"] = email
        
        return jsonify({"success": True, "message": "Google login successful"})
    except Exception as e:
        print(f"Google login error: {str(e)}")
        return jsonify({"error": str(e)}), 401

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if not auth:
        flash("Authentication service unavailable", "warning")
        return render_template("signup.html", is_signup_page=True)
        
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            auth.create_user_with_email_and_password(email, password)
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Signup failed: {str(e)}", "danger")
            return redirect(url_for("signup"))
    return render_template("signup.html", is_signup_page=True)

@app.route("/google-signup", methods=["POST"])
def google_signup():
    if not auth:
        return jsonify({"error": "Authentication service unavailable"}), 503
        
    try:
        # Get the ID token from the request
        id_token = request.json.get("idToken")
        if not id_token:
            return jsonify({"error": "No ID token provided"}), 400
            
        # Verify the ID token with Firebase
        decoded_token = auth.verify_id_token(id_token)
        
        # Get user info
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        
        # Store user info in session
        session["user"] = id_token
        session["user_email"] = email
        
        return jsonify({"success": True, "message": "Google signup successful"})
    except Exception as e:
        print(f"Google signup error: {str(e)}")
        return jsonify({"error": str(e)}), 401

@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("user_email", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

# Health check route for Railway
@app.route("/health")
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    # Railway deployment configuration
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    print(f"🚀 Starting Flask app on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)