import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import random

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# Load models and tokenizer with caching
@st.cache_resource
def load_models():
    # Load the trained models
    rnn_model = tf.keras.models.load_model("RNN_model.h5")
    lstm_model = tf.keras.models.load_model("LSTM_model.h5")
    bilstm_model = tf.keras.models.load_model("BiLSTM_model.h5")

    # Load the tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Define label_map directly
    label_map = {"positive": 2, "neutral": 1, "negative": 0}
    label_map_reverse = {v: k for k, v in label_map.items()}

    return rnn_model, lstm_model, bilstm_model, tokenizer, label_map_reverse


rnn_model, lstm_model, bilstm_model, tokenizer, label_map_reverse = load_models()


# Preprocessing function
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)


# --- Sample Reviews for Testing ---
sample_reviews = {
    "positive": [
        "This product is absolutely amazing! I'm so happy with my purchase. Highly recommend.",
        "Fantastic movie, truly captivating from start to finish. A must-watch!",
        "The customer service was exceptional. They went above and beyond to help me.",
        "Delicious food and a wonderful ambiance. I'll definitely be back soon.",
        "Seamless experience. Everything worked perfectly and easily.",
    ],
    "negative": [
        "Absolutely terrible. The worst product I've ever bought. Don't waste your money.",
        "Disappointing film, very boring and predictable. Wish I hadn't seen it.",
        "Rude and unhelpful staff. My issue was not resolved at all.",
        "The food was bland and the service was incredibly slow. Never again.",
        "Frustrating experience. Constantly crashing and full of bugs.",
    ],
    "neutral": [
        "The product arrived on time and was as described.",
        "The movie was okay, nothing special but not bad either.",
        "Customer service answered my question, but it took a while.",
        "The restaurant has a decent menu, but the prices are a bit high.",
        "The software performs its basic functions.",
    ],
}


# Streamlit UI
st.title("Review Sentiment Analysis")

selected_models = st.multiselect("Select models to use:", ["RNN", "LSTM", "BiLSTM"])

st.write("### Try a Sample Review")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Positive", key="positive_sample"):
        st.session_state.review_text = random.choice(sample_reviews["positive"])
with col2:
    if st.button("Negative", key="negative_sample"):
        st.session_state.review_text = random.choice(sample_reviews["negative"])
with col3:
    if st.button("Neutral", key="neutral_sample"):
        st.session_state.review_text = random.choice(sample_reviews["neutral"])

# Initialize session state for review_text if it doesn't exist
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

review = st.text_area(
    "Enter the review:", value=st.session_state.review_text, height=150
)

if st.button("Predict"):
    if not selected_models:
        st.warning("Please select at least one model.")
    elif not review.strip():
        st.warning("Please enter a review.")
    else:
        cleaned_review = preprocess_text(review)
        sequence = tokenizer.texts_to_sequences([cleaned_review])
        padded_sequence = pad_sequences(sequence, maxlen=100)

        predictions = {}
        for model_name in selected_models:
            if model_name == "RNN":
                model = rnn_model
            elif model_name == "LSTM":
                model = lstm_model
            elif model_name == "BiLSTM":
                model = bilstm_model
            probs = model.predict(padded_sequence)[0]
            predicted_class = np.argmax(probs)
            sentiment = label_map_reverse[predicted_class]
            predictions[model_name] = (sentiment, probs)

        st.write("### Model Predictions")

        # Highlighted Predictions
        cols = st.columns(len(selected_models))
        for i, (model_name, (sentiment, _)) in enumerate(predictions.items()):
            with cols[i]:
                st.metric(label=model_name, value=sentiment)

        # Create DataFrame for detailed scores
        prediction_data = []
        for model_name, (sentiment, probs) in predictions.items():
            pred_dict = {
                "Model": model_name,
                "Predicted Sentiment": sentiment,
                "Negative Score": f"{probs[0]:.4f}",
                "Neutral Score": f"{probs[1]:.4f}",
                "Positive Score": f"{probs[2]:.4f}",
            }
            prediction_data.append(pred_dict)

        df_predictions = pd.DataFrame(prediction_data).set_index("Model")

        # Display predictions table
        st.write("### Detailed Scores")
        st.table(df_predictions)

        # Check for consensus
        sentiments = [pred[0] for pred in predictions.values()]
        if len(set(sentiments)) == 1:
            st.success(f"All selected models agree on the sentiment: **{sentiments[0]}**.")
        else:
            st.warning("There is disagreement among the selected models.")