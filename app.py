import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}
reverse_word_index


model = load_model('imdb_sent_analysis')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in sample_review])

def preprocess_text(text):
    words =text.lower().split()
    enccoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([enccoded_review],maxlen = 500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction>0.5 else 'Negative'
    return sentiment,prediction

import streamlit as st
st.write('IMDB Movie Review Analysis')
st.write('Entet a movie review to classify it as positive or negative')
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction>0.5 else 'Negative'
else:
    st.write('Please enter a movie review')


