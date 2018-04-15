import time, math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score



def create_data_2(filename="train"):
    data = np.genfromtxt("dataset/adaboost_{}.csv".format(filename), delimiter=",")
    data = data[1:] # remove meta line
    features = []
    classes = []
    # data is a 2d array,
    for case in data:
        features.append(case[2:])
        classes.append(case[1:2])

    return features, classes


def adaboost_train(num_iterations, sample_weights=None, plot=True, show_accuracy=False):
    trainingset, trainingset_classes = create_data_2("train")
    iterationsAndErrors_trainingset = []

    num_cases = len(trainingset)
    init_weight = 1 / num_cases
    weights = np.full(num_cases, init_weight)

    classifiersAndAlphas = []

    for t in range(1, num_iterations+1):
        #  create a new- and learn a classifier ft: R --> {-1, +1}, using distribution weights
        dTree = DecisionTreeClassifier(max_depth=1)
        dTree.fit(trainingset, trainingset_classes, sample_weight=weights)

        predictions = dTree.predict(trainingset)
        # find error
        error = float(0)
        for i in range(num_cases):
            if predictions[i] != trainingset_classes[i]:
                error += weights[i]

        # find alpha
        alpha = 0.5 * np.log((1-error) / error)

        # update the weight distribution over examples
        # we need Z for this
        Zt = float(0)
        for i in range(num_cases):
            Zt += weights[i] * math.exp(-alpha * trainingset_classes[i] * predictions[i])
        # print("Zt {}".format(Zt))

        # now we can update the weights
        for i in range(num_cases):
            weights[i] = (weights[i] * math.exp(-alpha * trainingset_classes[i] * predictions[i])) / Zt

        # save the classifier
        classifiersAndAlphas.append((alpha, dTree))

        if show_accuracy:
            print("#{} -- Accuracy: {}".format(t, accuracy_score(trainingset_classes, predictions)))

        if plot:
            voted = vote_entire_dataset(classifiersAndAlphas, trainingset)
            training_error = adaboost_error(trainingset_classes, voted, display=False)
            iterationsAndErrors_trainingset.append((t, training_error))


    if plot:
        draw_error_rate(iterationsAndErrors_trainingset, legend="Training")

    # return all the weak classifiers, so they can vote
    return classifiersAndAlphas

def adaboost_test(classifiersAndAlphas, plot=True):
    testset, testset_classes = create_data_2("test")
    iterationsAndErrors_testset = []

    if plot:

        for t in range(1, len(classifiersAndAlphas)+1):
            voted = vote_entire_dataset(classifiersAndAlphas[:t], testset)
            testing_error = adaboost_error(testset_classes, voted)
            iterationsAndErrors_testset.append((t, testing_error))

        draw_error_rate(iterationsAndErrors_testset, legend="Test")

    voted = vote_entire_dataset(classifiersAndAlphas, testset)
    error = adaboost_error(testset_classes, voted) * 100
    print("FINAL: CORRECT: {:.2f}% -- ERROR: {:.2f}%".format(100-error, error))





def draw_error_rate(itAndErr, legend=""):
    iterations= []
    train= []
    for it, err in itAndErr:
        iterations.append(it)
        train.append(err)

    plt.plot(iterations, train, label=legend)
    plt.draw()

def vote_entire_dataset(classifiers, testset):
    vote = 0
    for alpha, classifier in classifiers:
        vote += alpha * classifier.predict(testset)

    vote = np.sign(vote)
    return vote


def vote_one_case(classifiers, case):
    vote = 0
    for alpha, classifier in classifiers:
        vote += alpha * classifier.predict(case)

    vote = np.sign(vote)
    return vote



def adaboost_error(labels, voted, display=True):
    correct = 0
    for idx, label in enumerate(labels):
        if label == voted[idx]:
            correct += 1

    accuracy = correct / len(labels)
    error = 1 - accuracy

    if display:
        print("ADABOOST accuracy {} :: Error {}".format(accuracy, error))
    return error


def main():
    classifiersAndAlphas = adaboost_train(10, show_accuracy=True)
    adaboost_test(classifiersAndAlphas)

    plt.title("Error Graph")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Error rate %")
    plt.show()

main()


