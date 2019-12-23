from hazm import *
import csv
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import numpy as np

# reading persian text from file
persian_text = ""
tree = ET.parse('Persian.xml')
root = tree.getroot()
persian_documents_list = []
for page in root:
    page_text = ""
    for text in page[3]:
        txt = text.text
        if (txt != None):
            persian_text += txt
            page_text += txt
    persian_documents_list.append(page_text)

# reading english text from file
english_text = ""
english_document_list = []
with open('English.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        english_document_list.append(row)
        english_text += " ".join(row)
    # print(english_document_list)


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


def pre_process_persian(persian_text):
    # normalizing
    normalizer = Normalizer()
    persian_text = normalizer.normalize(persian_text);

    # removing punctuations
    punctuations = '.!?,\'\"()-:;#][}{*=|\\'
    for mark in punctuations:
        persian_text = persian_text.replace(mark, " ")

    persian_text = ''.join(i for i in persian_text if not i.isdigit())

    # tokenizing
    tokens = word_tokenize(persian_text)

    # removing stopwords

    stop_words = stopwords_list()
    tokens = [w for w in tokens if not w in stop_words]

    # token_count = dict()
    # for token in tokens:
    #     if token in token_count:
    #         token_count[token] += 1
    #     else:
    #         token_count[token] = 1
    #
    # stop_words = []
    # for token in tokens:
    #     if token_count[token] > len(tokens) / 100:
    #         if token not in stop_words:
    #             stop_words.append(token)
    # print(stop_words)
    #
    # tokens = [token for token in tokens if token not in stop_words]
    # print(tokens)

    # lemmatizing
    lemmatized_tokens = []
    lemmatizer = Lemmatizer()
    for token in tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(token))

    # stemming
    stemmed_tokens = []
    stemmer = Stemmer()
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens



# print(pre_process_persian(persian_text[0:10000]))



