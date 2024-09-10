import streamlit as st 
import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib 

#same formart used during model training
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="text-classifier", layout="wide")

st.title("Text clasiffier application")

#Load downloaded keras model from google collab
text_model = tf.keras.models.load_model('text_model.keras')
#Load vectorizer model to convert the text into vectors for ML inference
vectorizer = joblib.load('vectorizer.pkl')

# Reuse function for preprocessing text during model training.
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  
    tokens = word_tokenize(text)
    lemmatized_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(lemmatized_tokens)

max_len = 79
vocab_size = 2389

input_text = st.text_area("Enter a text message:")


if st.button("Classify"):
    if input_text:
        preprocessed_text = preprocess_text(input_text)

        tfidf_vector = vectorizer.transform([preprocessed_text])

        
        tfidf_vector = tfidf_vector.toarray()

        prediction_proba = text_model.predict(tfidf_vector)

        ham_probability = prediction_proba[0][0]  
        spam_probability = prediction_proba[0][1]  

        
        predicted_label = "Spam" if spam_probability > ham_probability else "Ham"

        st.write(f"Prediction: {predicted_label}")
        st.write(f"Ham: {ham_probability * 100:.2f}% confidence")
        st.write(f"Spam: {spam_probability * 100:.2f}% confidence")
    else:
        st.write("Please enter a message to classify.")

