import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

#mapping of words back to letters

word_index=imdb.get_word_index()
rev={value:key for key,value in word_index.items()}

# Load the model

model=load_model('simplernnmodelfinal.h5')

# Function to DECODE REVIEW
def decodedreview(encodedreview):
    return [rev.get(i-3,'?') for i in encodedreview]

# Function to TAKE USER INPUT AND PROCESS IT

def preprocess_text(text):
    words = text.lower().split()
    
    # ENCODE THE REVIEW FROM THE DICTIONARY
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

###PREDICTION FUNCTION

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

### STREAMLIT APP
import streamlit as st

st.title("IMDB MOVIE REVIEW  - SENTIMENT ANALYSIS MODEL")
st.write(" TYPE ANY MOVIE REVIEW "
         "THIS MODEL WILL CLASSIFY IT AS POSITIVE OR NEGATIVE") 


userinput=st.text_area('Type movie review')

if st.button('Classify Now'):
    sentiment,score=predict_sentiment(userinput)

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')




