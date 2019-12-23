from hazm import *
import csv
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import numpy as np


def spell_correction_english(input_sentence):
    input_tokens = nltk.word_tokenize(input_sentence)
    corrected = ""
    for token in input_tokens:
        bg_input_word = set(nltk.ngrams(token, n=2))
        words = nltk.corpus.words.words()
        min_jaccard = 5;
        for word in words:
            bg_word = set(nltk.ngrams(word, n=2))
            jd = nltk.jaccard_distance(bg_input_word, bg_word)
            if jd < min_jaccard:
                suggested_words = []
                min_jaccard = jd
                suggested_words.append(word)
            if jd == min_jaccard:
                suggested_words.append(word)
        print(suggested_words)
        min_edit_distance = 1000
        return_words = []
        for word in suggested_words:
            ed = nltk.edit_distance(token, word)
            if ed < min_edit_distance:
                return_words = []
                min_edit_distance = ed
                return_words.append(word)
            if ed == min_edit_distance:
                return_words.append(word)

        corrected = corrected + " " + return_words[0]
    return corrected


def spell_correction_persian(input_sentence):
    input_tokens = word_tokenize(input_sentence)
    corrected = ""
    for token in input_tokens:
        bg_input_word = bi_gram(token)
        words = words_list()
        min_jaccard = 5;
        for word in words:
            word = word[0]
            bg_word = bi_gram(word)
            jd = jaccard_distance(bg_input_word, bg_word)
            if jd < min_jaccard:
                suggested_words = []
                min_jaccard = jd
                suggested_words.append(word)
            if jd == min_jaccard:
                suggested_words.append(word)
        print(suggested_words)
        min_edit_distance = 1000
        return_words = []
        for word in suggested_words:
            ed = edit_distance(token, word)
            if ed < min_edit_distance:
                return_words = []
                min_edit_distance = ed
                return_words.append(word)
            if ed == min_edit_distance:
                return_words.append(word)
        print(return_words)
        corrected = corrected + " " + return_words[0]
    return corrected


def bi_gram(token):
    bi_grams = []
    for i in range(len(token) - 1):
        bi_grams.append(token[i:i + 2])
    return bi_grams


def jaccard_distance(token1, token2):
    bi_gram_1 = bi_gram(token1)
    bi_gram_2 = bi_gram(token2)
    intersection = len([value for value in bi_gram_1 if value in bi_gram_2])
    union = len(bi_gram_1) + len(bi_gram_2) - intersection
    return 1 - (intersection / union)


def edit_distance(token1, token2):
    if len(token1) < len(token2):
        return edit_distance(token2, token1)

    if len(token2) == 0:
        return len(token1)

    previous_row = range(len(token2) + 1)
    for i, c1 in enumerate(token1):
        current_row = [i + 1]
        for j, c2 in enumerate(token2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]