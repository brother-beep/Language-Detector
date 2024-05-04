import streamlit as st 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import pyttsx3

lists = ['Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
       'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
       'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish']

def clean_text(text):
    text = re.sub(r'[!@#$%(),\n"?~`0-9]',' ',text)
    text = re.sub(r'[[ ]]',' ',text)
    text = text.lower()
    return text

# Load the trained model and TF-IDF vectorizer
with open('nb.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_tokenizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
    

def speak(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    # Convert text to speech
    engine.say(text)
    # Wait for speech to finish
    engine.runAndWait()


def predict_language(text):
    text = clean_text(text)
    vector = tfidf_vectorizer.transform([text])
    test = model.predict(vector)
    return lists[test[0]]

def main():
    st.title("Language Detector")
    
    # Text input area for user input
    text_input = st.text_area("Enter text for language detection:")
    
    if st.button("Detect Language"):
            predicted_language = predict_language(text_input)
            write = st.write(f"The Language is {predicted_language}")
            

if __name__ == "__main__":
    main()
