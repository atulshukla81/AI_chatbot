import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import os

lemmatizer = WordNetLemmatizer()

# Absolute paths to the .pkl files
base_path = "F:/Python Projects/chatbot_project/data"
words_path = os.path.join(base_path, "words.pkl")
classes_path = os.path.join(base_path, "classes.pkl")

# Load words and classes from .pkl files
with open(words_path, "rb") as f:
    words = pickle.load(f)
with open(classes_path, "rb") as f:
    classes = pickle.load(f)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

