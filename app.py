import streamlit as st
import nltk

# Check if 'punkt' tokenizer is available, if not download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


import pickle
import sklearn
import string
from nltk .corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    # breaking into separate words
    text = nltk.word_tokenize(text)
    
    # as text is converted to list after tokenization- so useing loop
    y=[]
    for i in text:
        if i.isalnum():   # just include alphanumeric- remove special characters
            y.append(i)
            
    text=y[:]   # removing stopwords and punctuation
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
            
    text=y[:]    # stemming  dancing-> danc, loving-> love
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


tfidf = pickle.load(open('vectorization.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip():  # make sure user typed something
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter a message to predict.")
