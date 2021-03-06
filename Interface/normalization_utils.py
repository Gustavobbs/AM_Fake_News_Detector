import numpy as np #importa a biblioteca usada para trabalhar com vetores de matrizes
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes (dados em formato de tabela) e análise de dados
import re #Regexr
import nltk
from nltk.stem import RSLPStemmer #Copyright (C) 2001-2019 NLTK Project
from nltk.metrics import BigramAssocMeasures
import glob
from scipy import sparse
import matplotlib.pyplot as plt

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
    return sparse.csr_matrix(result)

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

def get_feature_names(corpus, file_name='data/bow/common/feature_names.txt'):
    try:
        feature_names = [line.rstrip('\n') for line in open(file_name, 'r')]
        return feature_names
    except IOError:
        feature_names = []
        for sentence in corpus:
            tokens = sentence_tokenizer(sentence_preprocessor(sentence))
            feature_names.extend(tokens)
            feature_names = list(set(feature_names))

        f = open(file_name, 'w')
        for feature in feature_names:
            f.write("%s\n" % feature)
        f.close()

        return sorted(feature_names)

def bow_matrix(corpus, feature_names, file_name='data/bow/common/bow_matrix.npz'):
    try:
        X = sparse.load_npz(file_name)
        return X
    except IOError:
        X = sentence_to_features(corpus[0], feature_names)
        for i in range(1, len(corpus)):
            print('%d/%d' %(i, len(corpus)), end='\r', flush=True)
            X = sparse.vstack([X, sentence_to_features(corpus[i], feature_names)])
        sparse.save_npz(file_name, X)
        return X

def get_classes(quant_true_news, quant_fake_news):
    try:
        Y = np.load('data/bow/bow_classes.npy')
        return Y
    except IOError:
        Y = np.concatenate((np.zeros(quant_true_news, dtype=int), np.ones(quant_fake_news, dtype=int)), axis=None)
        np.save('data/bow/bow_classes.npy', Y)
        return Y

def build_bow():
    corpus, quant_true_news, quant_fake_news = get_news()

    print('Montando o BOW...')
    feature_names = get_feature_names(corpus)

    print('Montando a matriz...')
    X = bow_matrix(corpus, feature_names)

    Y = get_classes(quant_true_news, quant_fake_news)

    print('Quantidade de features:', len(feature_names))
    print('10 primeiros valores de Y:', Y[0:10])
    print('10 ultimos valores de Y:', Y[-10:])

    return X, Y, feature_names

def doc_features(document, features_list):
    features = []
    for word in features_list:
        features.append(1 if (word in document) else 0)
    return features

def build_reduced_bow_matrix(corpus, reduced_bow_file_name):
    try:
        X = sparse.load_npz(reduced_bow_file_name)
        return X
    except IOError:
        document_list = get_all_words(corpus)[1]
        X = sparse.csr_matrix([doc_features(document, features_list) for document in document_list])
        sparse.save_npz(X, reduced_bow_file_name)
        return X

def get_all_words(corpus):
    all_words = []
    document_list = []
    for sentence in corpus:
        tokens = sentence_tokenizer(sentence_preprocessor(sentence))
        document_list.append(tokens)
        all_words.extend(tokens)
    return(all_words, document_list)

def get_best_words(corpus, quant_true_news, quant, best_words_file_name, best_frequency_file_name):
    try:
        best_words = [line.rstrip('\n') for line in open(best_words_file_name, 'r')]
        best_frequency = [tuple(line.rstrip('\n').split(',')) for line in open(best_frequency_file_name, 'r')]
        return best_words, best_frequency
    except IOError:
        word_fd = nltk.probability.FreqDist()
        label_word_fd = nltk.probability.ConditionalFreqDist()

        all_words_true = get_all_words(corpus[:quant_true_news])[0]
        all_words_fake = get_all_words(corpus[quant_true_news:])[0]

        for word in all_words_true:
            word_fd[word] += 1
            label_word_fd['true'][word] += 1

        for word in all_words_fake:
            word_fd[word] += 1
            label_word_fd['fake'][word] += 1

        true_word_count = sum(label_word_fd['true'].values())
        fake_word_count = sum(label_word_fd['fake'].values())
        total_word_count = true_word_count + fake_word_count

        word_scores = {}

        for word, freq in word_fd.items():
            true_score = BigramAssocMeasures.chi_sq(label_word_fd['true'][word],
                (freq, true_word_count), total_word_count)
            fake_score = BigramAssocMeasures.chi_sq(label_word_fd['fake'][word],
                (freq, fake_word_count), total_word_count)
            word_scores[word] = true_score + fake_score

        best_frequency = sorted(word_scores.items(), key=lambda t: t[1], reverse=True)[:quant]
        best_words = list(set([w for w, s in best_frequency]))

        f = open(best_words_file_name, 'w')
        q = open(best_frequency_file_name, 'w')
        for word, frequency in zip(best_words, best_frequency):
            f.write("%s\n" % word)
            q.write("%s,%d\n" % (frequency[0], frequency[1]))
        f.close()
        q.close()
        return best_words, best_frequency

def build_reduced_bow(quant, best_words_file_name, best_frequency_file_name, reduced_bow_file_name, isBoolean):
    corpus, quant_true_news, quant_fake_news = get_news()

    features_list, frequency_list = get_best_words(corpus, quant_true_news, quant, best_words_file_name, best_frequency_file_name)

    X = []

    if isBoolean:
        X = build_reduced_bow_matrix(corpus, reduced_bow_file_name)
    else:
        X = bow_matrix(corpus, features_list, reduced_bow_file_name)

    Y = get_classes(quant_true_news, quant_fake_news)

    print('Quantidade de features:', len(features_list))
    print('10 primeiros valores de Y:', Y[0:10])
    print('10 ultimos valores de Y:', Y[-10:])

    return X, Y, features_list, frequency_list

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

def get_confusion_matrix(Y_test, Y_pred):
    cm = np.zeros([2, 2], dtype=int)

    for i in range(len(Y_pred)):
        cm[int(Y_test[i]), int(Y_pred[i])] += 1

    return cm

def get_performance(Y_test, Y_pred):
    confusion_matrix = get_confusion_matrix(Y_test, Y_pred)

    quant_data = confusion_matrix.sum()
    quant_classes = 2

    vp=np.zeros(quant_classes)
    vn=np.zeros(quant_classes)
    fp=np.zeros(quant_classes)
    fn=np.zeros(quant_classes)

    revocacao = np.zeros( quant_classes )
    revocacao_macroAverage = 0.0
    revocacao_microAverage = 0.0

    precisao = np.zeros( quant_classes )
    precisao_macroAverage = 0.0
    precisao_microAverage = 0.0

    fmedida = np.zeros( quant_classes )
    fmedida_macroAverage = 0.0
    fmedida_microAverage = 0.0

    for i in range(quant_classes):
        vp[i] = confusion_matrix[i, i]
        fp[i] = confusion_matrix[:, i].sum() - vp[i]
        fn[i] = confusion_matrix[i].sum() - vp[i]
        vn[i] = quant_data - vp[i] - fp[i] - fn[i]

    acuracia = confusion_matrix.diagonal().sum() / quant_data

    revocacao = vp / (vp + fn)
    revocacao_macroAverage = revocacao.sum() / quant_classes
    revocacao_microAverage = vp.sum() / (vp + fn).sum()

    precisao = vp / (vp + fp)
    precisao_macroAverage = precisao.sum() / quant_classes
    precisao_microAverage = vp.sum() / (vp + fp).sum()

    fmedida = 2 * ((precisao * revocacao) / (precisao + revocacao))
    fmedida_macroAverage = 2 * ((precisao_macroAverage * revocacao_macroAverage) / (precisao_macroAverage + revocacao_macroAverage))
    fmedida_microAverage = 2 * ((precisao_microAverage * revocacao_microAverage) / (precisao_microAverage + revocacao_microAverage))

    resultados = {'revocacao': revocacao, 'acuracia': acuracia, 'precisao': precisao, 'fmedida':fmedida}
    resultados.update({'revocacao_macroAverage':revocacao_macroAverage, 'precisao_macroAverage':precisao_macroAverage, 'fmedida_macroAverage':fmedida_macroAverage})
    resultados.update({'revocacao_microAverage':revocacao_microAverage, 'precisao_microAverage':precisao_microAverage, 'fmedida_microAverage':fmedida_microAverage})
    resultados.update({'confusionMatrix': confusion_matrix})

    return resultados

def learning_curve(X, Y, Xval, Yval, train, prediction):
    perf_train = []
    perf_val = []

    for i in range(10, len(Y)):
        print('%d/%d' %(i, len(Y)), end='\r', flush=True)

        train_result = train(X[0:i], Y[0:i])

        Y_pred_train = prediction(X[0:i], train_result)
        Y_pred_val = prediction(Xval, train_result)

        Y_train_acc = get_performance(Y[0:i], Y_pred_train)['acuracia']
        Y_val_acc = get_performance(Yval, Y_pred_val)['acuracia']

        perf_train.append(Y_train_acc)
        perf_val.append(Y_val_acc)

    # Define o tamanho da figura
    plt.figure(figsize=(20,12))

    # Plota os dados
    plt.plot(perf_train, color='blue', linestyle='-', linewidth=1.5, label='Treino')
    plt.plot(perf_val, color='red', linestyle='-', linewidth=1.5, label='Validação')

    # Define os nomes do eixo x e do eixo y
    plt.xlabel(r'# Qtd. de dados de treinamento',fontsize='x-large')
    plt.ylabel(r'Acuracia',fontsize='x-large')

    # Define o título do gráfico
    plt.title(r'Curva de aprendizado', fontsize='x-large')

    # Acrescenta um grid no gráfico
    plt.grid(axis='both')

    # Plota a legenda
    plt.legend()

    plt.savefig('output.png', bbox_inches='tight')

    plt.show()

def learning_curve_neural_networks(X, Y, Xval, Yval, vLambda, train, prediction):
    perf_train = []
    perf_val = []

    for i in range(10, len(Y)):
        print('%d/%d' %(i, len(Y)), end='\r', flush=True)

        Theta1, Theta2 = train(X[0:i], Y[0:i], vLambda)

        Y_pred_train = prediction(Theta1, Theta2, X[0:i])
        Y_pred_val = prediction(Theta1, Theta2, Xval)

        Y_train_acc = get_performance(Y[0:i], Y_pred_train)['acuracia']
        Y_val_acc = get_performance(Yval, Y_pred_val)['acuracia']

        perf_train.append(Y_train_acc)
        perf_val.append(Y_val_acc)

    # Define o tamanho da figura
    plt.figure(figsize=(20,12))

    # Plota os dados
    plt.plot(perf_train, color='blue', linestyle='-', linewidth=1.5, label='Treino')
    plt.plot(perf_val, color='red', linestyle='-', linewidth=1.5, label='Validação')

    # Define os nomes do eixo x e do eixo y
    plt.xlabel(r'# Qtd. de dados de treinamento',fontsize='x-large')
    plt.ylabel(r'Acuracia',fontsize='x-large')

    # Define o título do gráfico
    plt.title(r'Curva de aprendizado', fontsize='x-large')

    # Acrescenta um grid no gráfico
    plt.grid(axis='both')

    # Plota a legenda
    plt.legend()

    plt.show()

def get_U_and_S(X):
    m = X.shape[0]

    U, S = sparse.linalg.svds(sparse.csr_matrix(X, dtype=float), k=X.shape[0])[0:2]

    return U, S

def pca(X, K=0):
    U, S = get_U_and_S(X)
    if(K==0):
        n = X.shape[1]
        for k in range(n):
            if S[0:k].sum() / S.sum() >= 0.95:
                K = k
                break

    Z = np.zeros( [X.shape[0],K] )

    print(X.shape)
    print(U.shape)

    Z = X.dot(U[:, 0:K])

    return Z


download_punkt()
download_rslp()
download_stopwords()
