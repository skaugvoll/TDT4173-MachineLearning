from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout(pad=2.0)
    plt.ylabel('Target test data')
    plt.xlabel('Predicted values')

def run(classifier, random_state):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=random_state)

    if classifier == "knn":
        # k-nn
        neigh = KNeighborsClassifier()
        neigh.fit(X_train, y_train)
        y_predicted = neigh.predict(X_test)
        cm = confusion_matrix(y_test, y_predicted)
        plot_confusion_matrix(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],title="k-NN")
        plt.show()
    if classifier == "svm":
        #SVM
        svm_classifier = svm.SVC(kernel='linear')
        svm_classifier.fit(X_train, y_train)
        y_predicted = svm_classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_predicted)
        plot_confusion_matrix(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],title="SVM")
        plt.show()
    if classifier == "rtrees":
        #Random forest
        clf = RandomForestClassifier()
        clf = clf.fit(X_train,y_train)
        y_predicted = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_predicted)
        plot_confusion_matrix(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],title="Random Forest")
        plt.show()


run("svm", 101)
run("knn",101)
run("rtrees",101)

