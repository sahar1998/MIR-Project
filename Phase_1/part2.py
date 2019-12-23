import part1
import io


def pos_index_english(docs):
    positional_index_english = {}
    doc_id = 0
    for doc in docs:
        for pos, term in enumerate(part1.pre_process_english(doc[0] + " " + doc[1])):
            if term in positional_index_english:
                positional_index_english[term][0] += 1
                if doc_id in positional_index_english[term][1]:
                    positional_index_english[term][1][doc_id].append(pos)
                else:
                    positional_index_english[term][1][doc_id] = [pos]
            else:
                positional_index_english[term] = []
                positional_index_english[term].append(1)
                positional_index_english[term].append({})
                positional_index_english[term][1][doc_id] = [pos]
        doc_id += 1
    return positional_index_english


def pos_index_persian(docs):
    positional_index_persian = {}
    doc_id = 0
    for doc in docs:
        for pos, term in enumerate(part1.pre_process_persian(doc)):
            if term in positional_index_persian:
                positional_index_persian[term][0] += 1
                if doc_id in positional_index_persian[term][1]:
                    positional_index_persian[term][1][doc_id].append(pos)
                else:
                    positional_index_persian[term][1][doc_id] = [pos]
            else:
                positional_index_persian[term] = []
                positional_index_persian[term].append(1)
                positional_index_persian[term].append({})
                positional_index_persian[term][1][doc_id] = [pos]
        doc_id += 1
    return positional_index_persian


# bigram index
def find_bigrams(term):
    bigrams = []
    for i in range(len(term) - 1):
        bigrams.append(term[i:i + 2])
    return bigrams


def bigram_index_english(docs):
    persian_bigram_index = {}
    for doc in docs[0:10]:
        for term in part1.pre_process_persian(doc):
            for bigram in find_bigrams(term):
                if bigram in persian_bigram_index:
                    if term not in persian_bigram_index[bigram]:
                        persian_bigram_index[bigram].append(term)
                else:
                    persian_bigram_index[bigram] = [term]
    return persian_bigram_index


def bigram_index_english(docs):
    english_bigram_index = {}
    for doc in docs:
        for term in part1.pre_process_persian(doc):
            for bigram in find_bigrams(term):
                if bigram in english_bigram_index:
                    if term not in english_bigram_index[bigram]:
                        english_bigram_index[bigram].append(term)
                else:
                    english_bigram_index[bigram] = [term]
    return english_bigram_index


# pos_index_en = pos_index_english(part1.english_document_list)
def file_write(name, dict):
    f = open(name, "w")
    f.write(str(dict))
    f.close()


def file_write_persian(name, dict):
    f = open(name, "w", encoding="utf-8")
    f.write(str(dict))
    f.close()


def file_read(name):
    data = eval(open(name, 'r').read())
    return data


def file_read_persian(name):
    data = eval(open(name, 'r').read().decode('utf8'))
    return data


# pos_index_per = pos_index_persian(part1.persian_documents_list[0])
# file_write_persian('positional_index_persian.txt', pos_index_per)
file_write('bigram_index_english.txt', bigram_index_english(pos_index_english(part1.english_document_list)))
term = input("enter the word:\n")
pos_index_en = file_read('positional_index_english.txt')
# print(pos_index_en[term])
