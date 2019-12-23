from classification import Naive_bayesian, KNN, random_forest, SVM
from sklearn.metrics import accuracy_score, classification_report
from classification import test_document_list, train_document_list
from search import Naive_bayesian

classes_test = [test_document[0] for test_document in test_document_list]
classes_pred_1 = Naive_bayesian(train_document_list[1:500], test_document_list[1:20])
# print(classes_test, classes_pred_1)
print(classification_report(classes_test[1:20],classes_pred_1))
print(accuracy_score(classes_test[1:20], classes_pred_1))

classes_pred_2 = KNN(train_document_list[1:200], test_document_list[1:20], 5)
# print(classes_test[1:20], classes_pred_2)
print(classification_report([int(c) for c in classes_test[1:20]],classes_pred_2))
print(accuracy_score(classes_test[1:20], classes_pred_2))

classes_pred_3 = SVM(train_document_list[1:500], test_document_list[1:20])
print(classification_report(classes_test[1:20], classes_pred_3))
print(accuracy_score(classes_test[1:20], classes_pred_3))

classes_pred_4 = random_forest(train_document_list[1:500], test_document_list[1:20])
print(classification_report(classes_test[1:20],classes_pred_4))
print(accuracy_score(classes_test[1:20], classes_pred_4))