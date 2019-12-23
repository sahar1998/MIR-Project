from hazm import *
import csv

from nltk.classify import svm
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection, naive_bayes, svm
from Phase_1 import part5



# reading english text from file
train_text = ""
train_document_list = []
with open('phase2_train.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        train_document_list.append(row)
        train_text += " ".join(row)
    # print(english_document_list)
    # print(english_text)
test_text = ""
test_document_list = []
with open('phase2_test.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        test_document_list.append(row)
        test_text += " ".join(row)


def pre_process_english(english_text):
    # case folding
    english_text = english_text.lower()

    # removing punctuations
    punctuations = '.!?,\'\"()-:;#'
    for mark in punctuations:
        english_text = english_text.replace(mark, " ")
    # print(english_text)

    english_text = ''.join(i for i in english_text if not i.isdigit())

    # tokenizing english text
    tokens = word_tokenize(english_text)
    print(tokens)

    # removing stopwords : nltk stopwords, costume stopwords

    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    # token_count = dict()
    # for token in tokens:
    #     if token in token_count:
    #         token_count[token] += 1
    #     else:
    #         token_count[token]= 1
    #
    # stop_words = []
    # for token in tokens:
    #     if (token_count[token] > len(tokens)/300):
    #         if token not in stop_words:
    #             stop_words.append(token)
    # print(stop_words)
    #
    # tokens=[token for token in tokens if token not in stop_words]
    print(tokens)

    # stemming
    stemmed_tokens = []
    ps = PorterStemmer()
    for token in tokens:
        stemmed_tokens.append(ps.stem(token))

    print(stemmed_tokens)

    return stemmed_tokens


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
        idfDict[word] = math.log(N / float(val + 1))
    return idfDict


def compute_tfidf(document_dict, idfs):
    tfidf = {}
    for word, val in document_dict.items():
        tfidf[word] = val * idfs[word]
    return tfidf



def Naive_bayesian (train_document_list, test_document_list):

    World_class = []
    World_class_text = ""
    Sports_class= []
    Sports_class_text = ""
    Buisiness_class = []
    Buisiness_class_text = ""
    Sci_Tech_class =[]
    Sci_Tech_class_text = ""

    for train_document in train_document_list:
        if (train_document[0] == '1'):
            World_class.append(train_document)
            World_class_text += (train_document[2]+ " ")
        if (train_document[0] == '2'):
            Sports_class.append(train_document)
            Sports_class_text += (train_document[2] + " ")
        if (train_document[0] == '3'):
            Buisiness_class.append(train_document)
            Buisiness_class_text += (train_document[2] + " ")
        if (train_document[0] == '4'):
            Sci_Tech_class.append(train_document)
            Sci_Tech_class_text += (train_document[2] + " ")

    world_list = World_class_text.split(" ")
    sport_list = Sports_class_text.split(" ")
    business_list = Buisiness_class_text.split(" ")
    sci_tech_list = Sci_Tech_class_text.split(" ")

    unique_words = []
    for document in train_document_list:
        document_text = document[1] + " " + document[2]
        unique_words = set(unique_words).union(set(nltk.word_tokenize(document_text)))


    world_dict = {}
    sport_dict = {}
    business_dict = {}
    sci_tech_dict = {}
    for word in unique_words:
        world_dict[word] = 0
        for token in world_list:
            if word == token:
                world_dict[word] += 1
        world_dict[word] = (world_dict[word] + 1) / (len(world_list) + len(unique_words))

        sport_dict[word] = 0
        for token in sport_list:
            if word == token:
                sport_dict[word] += 1
        sport_dict[word] = (sport_dict[word] + 1) / (len(sport_list) + len(unique_words))

        business_dict[word] = 0
        for token in business_list:
            if word == token:
                business_dict[word] += 1
        business_dict[word] = (business_dict[word] + 1) / (len(business_list) + len(unique_words))

        sci_tech_dict[word] = 0
        for token in business_list:
            if word == token:
                sci_tech_dict[word] += 1
        sci_tech_dict[word] = (sci_tech_dict[word] + 1) / (len(sci_tech_list) + len(unique_words))

    classification_correct = 0
    classes = []
    for test_document in test_document_list:
        document_text = test_document[1] + " " + test_document[2]
        tokens = nltk.word_tokenize(document_text)
        p_world = len(World_class) / len(train_document_list)
        p_sport = len(Sports_class) / len(train_document_list)
        p_business = len(Buisiness_class) / len(train_document_list)
        p_sci_tech = len(Sci_Tech_class) / len(train_document_list)

        for token in tokens:
            if (token in unique_words):
                p_world = p_world * world_dict[token]
                p_sport = p_sport * sport_dict[token]
                p_business = p_business * business_dict[token]
                p_sci_tech = p_sci_tech * sci_tech_dict[token]
        p = [p_world, p_sport, p_business, p_sci_tech]
        count = [0 for _ in range(4)]
        count[p.index(max(p))] += 1
        if p.index(max(p)) + 1 == int(test_document[0]):
            classification_correct += 1
        classes.append(str(p.index(max(p)) + 1))
    print(classification_correct / len(test_document_list))
    return classes

Naive_bayesian(train_document_list[1:300], test_document_list[1:20])

def KNN(train_document_list, test_document_list, k):
    classes = []
    classification_correct = 0
    for test_document in test_document_list:
        knn = part5.compute_cosine_phase_2(test_document[2], train_document_list, k)
        # print(knn)
        count = [0 for _ in range(4)]
        for train_document in knn:
            count[int(train_document[0]) - 1] += 1
        # print(count.index(max(count)))
        classes.append(count.index(max(count)) + 1)
        # if count.index(max(count)) + 1 == int(test_document[0]):
        #     classification_correct +=1

    # print(classification_correct / len(test_document_list))
    return classes

# KNN(train_document_list[1:500], test_document_list[1:10], 9)

def SVM (train_document_list, test_document_list):
    train_document_dict_list = []
    unique_words = []
    for train_document in train_document_list:
        train_document_text = train_document[1] + " " + train_document[2]
        unique_words = set(unique_words).union(set(nltk.word_tokenize(train_document_text)))
    for test_document in test_document_list:
        test_document_text = test_document[1] + " " + test_document[2]
        unique_words = set(unique_words).union(set(nltk.word_tokenize(test_document_text)))

    for train_document in train_document_list:
        train_document_text = train_document[1] + " " + train_document[2]
        train_document_dict_list.append(make_dict(train_document_text, unique_words))
    train_idfs = compute_idf(train_document_dict_list, unique_words)

    train_x_tfidf = []
    train_y = []
    for train_document in train_document_list:
        train_document_text = train_document[1] + " " + train_document[2]
        train_x_tfidf.append(list(compute_tfidf(compute_tf(make_dict(train_document_text, unique_words), train_document_text), train_idfs).values()))
        train_y.append(train_document[0])

    test_x_tfidf = []
    test_y = []
    for test_document in test_document_list:
        test_tf = compute_tf(make_dict(test_document[2], set(nltk.word_tokenize(test_document[2])).union(unique_words)), test_document[2])
        test_idf = compute_idf(train_document_dict_list, set(nltk.word_tokenize(test_document[2])).union(unique_words))
        test_tfidf = compute_tfidf(test_tf, test_idf)
        test_x_tfidf.append(list(test_tfidf.values()))
        test_y.append(test_document[0])



    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_x_tfidf, train_y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(test_x_tfidf)
    # Use accuracy_score function to get the accuracy
    # print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, test_y) * 100)
    return predictions_SVM

# SVM(train_document_list[1:1000], test_document_list[1:100])

def random_forest (train_document_list, test_document_list):
    train_document_dict_list = []
    unique_words = []
    for train_document in train_document_list:
        train_document_text = train_document[1] + " " + train_document[2]
        unique_words = set(unique_words).union(set(nltk.word_tokenize(train_document_text)))
    for test_document in test_document_list:
        test_document_text = test_document[1] + " " + test_document[2]
        unique_words = set(unique_words).union(set(nltk.word_tokenize(test_document_text)))

    for train_document in train_document_list:
        train_document_text = train_document[1] + " " + train_document[2]
        train_document_dict_list.append(make_dict(train_document_text, unique_words))
    train_idfs = compute_idf(train_document_dict_list, unique_words)

    train_x_tfidf = []
    train_y = []
    for train_document in train_document_list:
        train_document_text = train_document[1] + " " + train_document[2]
        train_x_tfidf.append(list(
            compute_tfidf(compute_tf(make_dict(train_document_text, unique_words), train_document_text),
                          train_idfs).values()))
        train_y.append(train_document[0])

    test_x_tfidf = []
    test_y = []
    for test_document in test_document_list:
        test_tf = compute_tf(make_dict(test_document[2], set(nltk.word_tokenize(test_document[2])).union(unique_words)),
                             test_document[2])
        test_idf = compute_idf(train_document_dict_list, set(nltk.word_tokenize(test_document[2])).union(unique_words))
        test_tfidf = compute_tfidf(test_tf, test_idf)
        test_x_tfidf.append(list(test_tfidf.values()))
        test_y.append(test_document[0])


    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(train_x_tfidf, train_y)
    prediction = classifier.predict(test_x_tfidf)
    return prediction
    # print(accuracy_score(test_y, prediction))

# random_forest(train_document_list[1:700], test_document_list[1:20])