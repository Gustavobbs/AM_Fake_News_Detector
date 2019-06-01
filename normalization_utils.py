import numpy as np #importa a biblioteca usada para trabalhar com vetores de matrizes
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes (dados em formato de tabela) e análise de dados
import re #Regexr
import nltk
from nltk.stem import RSLPStemmer #Copyright (C) 2001-2019 NLTK Project

def download_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def download_rslp():
    try:
        nltk.data.find('stemmers/rslp')
    except LookupError:
        nltk.download('rslp')

def download_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def remove_symbols(sentence):
    return re.sub('[^a-z A-Z]', '', sentence) #mantém apenas letras e espaços

def sentence_to_lower(sentence):
    return sentence.lower()

def sentence_preprocessor(sentence):
    return remove_symbols(sentence_to_lower(sentence))

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stemming(tokenized_sentence):
    stemmer = RSLPStemmer()
    stemmed_tokenized_sentence = []
    for token in tokenized_sentence:
        stemmed_tokenized_sentence.append(stemmer.stem(token))
    return stemmed_tokenized_sentence

def remove_stop_words(tokenized_sentence):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    filterd_tokenized_sentence = []
    for token in tokenized_sentence:
        if token not in stopwords:
            filterd_tokenized_sentence.append(token)
    return filterd_tokenized_sentence

def sentence_tokenizer(sentence):
    return stemming(remove_stop_words(tokenize(sentence)))

download_punkt()
download_rslp()
download_stopwords()
