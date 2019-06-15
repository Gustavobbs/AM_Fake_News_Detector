import numpy as np
from naiveBayes import train
from normalization_utils import build_reduced_bow

X, Y, feature_names, frequency_list = build_reduced_bow(1000,
    'data/bow/reduced_binary/best_words_1000.txt',
    'data/bow/reduced_binary/best_frequency_1000.txt',
    'data/bow/reduced_binary/reduced_bow_1000.npz',
    True)

returnValues = np.array([train(X, Y)])

np.save('Interface/probabilities', returnValues)
