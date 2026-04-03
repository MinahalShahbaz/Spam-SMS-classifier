
import streamlit as st
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

input_sms = st.text_arae("Enter the message")

if st.button('predict'):

 # 1. preprocess
 transformed_sms = transform_text(input_sms) 
 # 2. vectorize
 vector_input = tfidf.transform([transformed_sms])
 # 3. predict
 result = model.predict(vector_input)[0]
 # 4. Display
if result == 1:
    st.header("Spam")
else:
    st.header("Spam Button")