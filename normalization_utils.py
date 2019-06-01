import numpy as np #importa a biblioteca usada para trabalhar com vetores de matrizes
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes (dados em formato de tabela) e análise de dados
import re #Regexr
import nltk
from nltk.stem import RSLPStemmer #Copyright (C) 2001-2019 NLTK Project
from sklearn.feature_extraction.text import CountVectorizer
import glob

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

def sentence_to_features(sentence, feature_names):
    tokenized_sentence = sentence_tokenizer(sentence_preprocessor(sentence))
    result = np.zeros(len(feature_names), dtype=int)
    for token in tokenized_sentence:
        if token in feature_names:
            result[feature_names.index(token)] += 1
    return result

def get_news():
    corpus = []

    print('Pegando notícias verdadeiras...')
    for file in glob.glob("Fake.br-Corpus/full_texts/true/*.txt"):
        news = open(file, "r", encoding = "unicode_escape")
        sentence = news.read().replace('\n',' ')
        corpus.append(sentence)
        news.close()

    quant_true_news = len(corpus)

    print('Pegando notícias falsas...')
    for file in glob.glob("Fake.br-Corpus/full_texts/fake/*.txt"):
        news = open(file, "r", encoding = "unicode_escape")
        sentence = news.read().replace('\n',' ')
        corpus.append(sentence)
        news.close()

    quant_fake_news = len(corpus) - quant_true_news

    print('Quantidade de notícias verdadeiras:', quant_true_news)
    print('Quantidade de notícias falsas:', quant_fake_news)

    return corpus, quant_true_news, quant_fake_news

def build_bow():
    cv = CountVectorizer(preprocessor=sentence_preprocessor, tokenizer=sentence_tokenizer)

    corpus, quant_true_news, quant_fake_news = get_news()

    print('Montando o BOW...')
    X = cv.fit_transform(corpus)
    feature_names = cv.get_feature_names()
    Y = np.concatenate((np.zeros(quant_true_news, dtype=int), np.ones(quant_fake_news, dtype=int)), axis=None)

    print('Quantidade de features:', len(feature_names))
    print('10 primeiros valores de Y:', Y[0:10])
    print('10 ultimos valores de Y:', Y[-10:])

    return X, Y, feature_names

def stratified_kfolds(Y, k):
    train_index = []
    test_index = []
    folds_final = []

    for i in range(k):
        train_index.append([])
        test_index.append([])
        folds_final.append([train_index[i],test_index[i]])

    classes_indexes = []
    classes_indexes.append([i for i, x in enumerate(Y==0) if x])
    classes_indexes.append([i for i, x in enumerate(Y==1) if x])

    for i in range(k):
        for class_indexes in classes_indexes:
            aux = class_indexes[0:int(len(class_indexes)*i/k)]
            aux2 = class_indexes[int(len(class_indexes)*(i+1)/k):len(class_indexes)]
            train_index[i].extend(aux + aux2)
            test_index[i].extend(class_indexes[int(len(class_indexes)*i/k):int(len(class_indexes)*(i+1)/k)])
        train_index[i].sort()
        test_index[i].sort()

    return folds_final

download_punkt()
download_rslp()
download_stopwords()
