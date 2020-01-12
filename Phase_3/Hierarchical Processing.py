from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import csv
import numpy as np

def vectorize(line):
    words = []
    for word in line: # line - iterable, for example list of tokens
        try:
            w2v_idx = w2v_indices[word]
        except KeyError: # if you does not have a vector for this word in your w2v model, continue
            continue
        words.append(w2v_vectors[w2v_idx])
        if words:
            words = np.asarray(words)
            min_vec = words.min(axis=0)
            max_vec = words.max(axis=0)
            return np.concatenate((min_vec, max_vec))
        if not words:
            return None


documents_list = []
documents_id_list = []
documents_words_list = []
with open('Data.csv', 'r', encoding="iso-8859-1") as f:
    reader = csv.reader(f)
    for row in reader:
        documents_list.append(row[1])
        documents_id_list.append(row[0])
        documents_words_list.append(row[1].split(" "))


W2V_model = Word2Vec(documents_words_list, min_count=1)
response2 = W2V_model[W2V_model.wv.vocab]
w2v_vectors = W2V_model.wv.vectors # here you load vectors for each word in your model
w2v_indices = {word: W2V_model.wv.vocab[word].index for word in W2V_model.wv.vocab}
response2= []
for document in documents_words_list:
    response2.append(vectorize(document))


vectorizer = TfidfVectorizer()
response1 = vectorizer.fit_transform(documents_list)
response1 = response1.toarray()

model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
model.fit(response1[0:100])
labels = model.labels_
with open('tfidf-HierProc.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "cluster"])
    for i in range(1,len(documents_list[0:100])):
        writer.writerow([documents_id_list[i], labels[i]])

model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
model.fit(response2)
labels = model.labels_
with open('W2V-HierProc.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "cluster"])
    for i in range(1,len(documents_list)):
        writer.writerow([documents_id_list[i], labels[i]])
