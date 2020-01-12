from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans
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

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 2)

    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()


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


true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(response1)
labels = model.labels_

with open('tfidf-KMeans.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "cluster"])
    for i in range(1,len(documents_list)):
        writer.writerow([documents_id_list[i], labels[i]])


true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(response2)
labels = model.labels_

clusters = [[] for i in range(len(documents_words_list))]
with open('W2V-KMeans.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "cluster"])
    for i in range(1,len(documents_words_list)):
        writer.writerow([documents_id_list[i], labels[i]])



