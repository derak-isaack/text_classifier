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

#Load downloaded keras model from google collab
text_model = tf.keras.models.load_model('my_model.keras')
#Load vectorizer model to convert the text into vectors for ML inference
vectorizer = joblib.load('tfidf_vectorizer.pkl')

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

#Define streamlit logic to classify input text. 
if st.button("Classify"):
    if input_text:
        preprocessed_text = preprocess_text(input_text)

        tfidf_vector = vectorizer.transform([preprocessed_text])

        prediction = text_model.predict(tfidf_vector)

        predicted_label = "Spam" if prediction[0][1] > 0.5 else "Ham"
        st.write(f"Prediction: {predicted_label}")
    else:
        st.write("Please enter a message to classify.")