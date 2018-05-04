from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from data_generator import DataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

def SVM(trainingset="../chars74k-lite", testingset="../detection-images"):
    dg = DataGenerator(dataset=trainingset, normalized=True)
    dg.shuffle_data()

    training_cases, training_labels = dg.get_training_partition(percentage=.8)
    testing_cases, testing_labels = dg.get_testing_partition(percentage=.2)

    classifier = svm.SVC(kernel='poly',gamma=0.05, degree=2,probability=False,C=1.0)
    #classifier = svm.SVC(kernel='linear', C=0.85)
    classifier.fit(training_cases,training_labels)
    y_predicted = classifier.predict(testing_cases)

    print(classifier)
    correct_count = 0
    for i in range(len(y_predicted)):
        if y_predicted[i] == testing_labels[i]:
            correct_count +=1

    print(correct_count / len(y_predicted))

    return

if __name__ == "__main__":
    SVM()