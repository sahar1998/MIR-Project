from hazm import *
import csv
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import numpy as np


# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(english_document_list)
# feature_names = vectorizer.get_feature_names()
# dense = vectors.todense()
# denselist = dense.tolist()


def make_dict(document, unique_words):
    document_dict = dict.fromkeys(unique_words, 0)
    for word in nltk.word_tokenize(document):
        document_dict[word] += 1
    return document_dict


def compute_tf(document_dict, document):
    tfDict = {}
    document_size = len(document)
    for word, count in document_dict.items():
        tfDict[word] = count / float(document_size)
    return tfDict


def compute_idf(documents, unique_words):
    import math
    N = len(documents)
    idfDict = dict.fromkeys(unique_words, 0)
    for document_dict in documents:
        for word, val in document_dict.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def compute_tfidf(document_dict, idfs):
    tfidf = {}
    for word, val in document_dict.items():
        tfidf[word] = val * idfs[word]
    return tfidf


test_document_list = ['the man went out for a walk', 'the children sat around the fire']


# print(english_document_list)


def compute_cosine(query, document_list):
    document_dict_list = []
    unique_words = []
    for document in document_list:
        document_text = document[0] + " " + document[1]
        unique_words = set(unique_words).union(set(nltk.word_tokenize(document_text)))
    unique_words = unique_words.union(set(nltk.word_tokenize(query)))

    for document in document_list:
        document_text = document[0] + " " + document[1]
        document_dict_list.append(make_dict(document_text, unique_words))

    query_tf = compute_tf(make_dict(query, set(nltk.word_tokenize(query))), query)
    query_idf = compute_idf(document_dict_list, set(nltk.word_tokenize(query)).union(unique_words))
    query_tfidf = compute_tfidf(query_tf, query_idf)

    idfs = compute_idf(document_dict_list, unique_words)

    cosine = []
    for document in document_list:
        document_text = document[0] + " " + document[1]
        similarity = 0
        tfidf = compute_tfidf(compute_tf(make_dict(document_text, unique_words), document_text), idfs)
        for word1 in query_tfidf:
            for word2 in tfidf:
                if (word1 == word2):
                    similarity += query_tfidf[word1] * tfidf[word2]
        cosine.append(similarity)

    answer = []
    k = 10
    cosine = np.array(cosine)
    ind = cosine.argsort()[-k:]  # index of the k highest elements
    for index in ind:
        answer.append(document_list[index])
    print(answer)