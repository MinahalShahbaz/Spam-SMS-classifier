import streamlit as st
import nltk
import pickle

# Ensure NLTK punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize

# Load model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def transform_text(text):
    tokens = word_tokenize(text)
    return " ".join(tokens)

st.title("Email/Spam Classifier")
input_sms = st.text_area("Enter your message:")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter a message before predicting!")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
