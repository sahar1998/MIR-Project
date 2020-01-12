import csv

import nltk

from . import part5



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
        document_text = test_document[0] + " " + test_document[1]
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
        classes.append(str(p.index(max(p)) + 1))
    return classes



train_text = ""
train_document_list = []
with open('phase2_train.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        train_document_list.append(row)
        train_text += " ".join(row)

english_text = ""
english_document_list = []
with open('English.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        english_document_list.append(row)
        english_text += " ".join(row)

def search_query(query, c, document_list):
    classified = Naive_bayesian(train_document_list[1:100], document_list[1:])
    documents_in_c = []
    for i in range(1, len(document_list)-1 ):
        if classified[i] == c:
            documents_in_c.append(document_list[i])
    answer = part5.compute_cosine(query, documents_in_c, 3)
    return answer

# print(search_query("fire a", '1', english_document_list))



