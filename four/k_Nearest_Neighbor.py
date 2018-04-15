
'''
kNN (dataset, sample){
   1. Go through each item in my dataset, and calculate the "distance"
   from that data item to my specific sample.
   2. Classify the sample as the majority class between K samples in
   the dataset having minimum distance to the sample.
}

k-NN regression
In k-NN regression, the k-NN algorithm is used for estimating continuous variables.
One such algorithm uses a weighted average of the k nearest neighbors,
weighted by the inverse of their distance. This algorithm works as follows:

1. Compute the Euclidean or Mahalanobis distance from the query example to the labeled examples.
2. Order the labeled examples by increasing distance.
3. Find a heuristically optimal number k of nearest neighbors, based on RMSE. This is done using cross validation.
4. Calculate an inverse distance weighted average with the k-nearest multivariate neighbors.

The Inverse Distance Weighting interpolator assumes that each input point
has a local influence that diminishes with distance.
It weights the points closer to the processing cell greater than those further away.
A specified number of points, or all points within a specified radius can be used
to determine the output value of each location.
Use of this method assumes the variable being mapped decreases in influence with distance
from its sampled location.
'''

def import_or_install_packages():
    global sys, operator, Q, euclidean
    import sys
    import operator
    ### EASY and performance queue for neighbors
    try:
        import Queue as Q  # python ver. < 3.0
    except ImportError:
        import queue as Q
    ### EASY euclidean distance
    try:
        from scipy.spatial.distance import euclidean
    except Exception as e:
        import pip
        pip.main(['install', "scipy"])
        from scipy.spatial.distance import euclidean



def create_data_set(filename="classification"):
    data = [] # list with tuples
    with open("dataset/knn_"+str(filename)+".csv", "r") as f:
        firstLine = f.readline() # discards the first line of the file. meta line
        for line in f:
            data.append( tuple(float(x) for x in line.split(",")) )
    return data


def calculate_euclidean_distance(sampleToClassify, case):
    if(len(case) == len(sampleToClassify)):
        return euclidean(list(sampleToClassify[:-1]), list(case[:-1])) # splice to remove class from distance
    elif(len(case) == len(sampleToClassify)+1):
        return euclidean(list(sampleToClassify), list(case[:-1])) # sampleToClassify do not have target / class given



def k_nearest_neighbor(sampleToClassify, k=2, regression=False, dataset="classification"):
    data = create_data_set(dataset)
    nearest_neighbors = Q.PriorityQueue()

    # TODO step 1: find neighbors
    for case in data:
            nearest_neighbors.put((calculate_euclidean_distance(sampleToClassify, case), case))

    # TODO step 2: find classification based on k neighbors
    classes = {}
    for n in range(0, k):
        nn = nearest_neighbors.get() # get nearest neighbor
        print("nn", nn)
        c = nn[1][-1]  # get case and its target

        if c in classes:
            classes[c][0] += 1
            classes[c][1].append(nn[0])

        else:
            classes[c] = [1, [nn[0]]]

    if regression:
        weighted_classes = {}
        for key_value in classes.keys():
            weighted_classes[key_value] = 0
            for dist in classes[key_value][1]:
                if(dist == 0):
                    dist = 1
                weighted_classes[key_value] += 1 / dist
            weighted_classes[key_value] /= len(classes[key_value][1])
        return max(weighted_classes.items(), key=operator.itemgetter(1))[0]
    else:
        return max(classes.items(), key=operator.itemgetter(1))[0]



if __name__ == "__main__":
    import_or_install_packages()
    example_sampleToClassify = (6.3, 2.7, 4.9, 1.8) # 123 example: as specified in assignment text is actually line 125
    # REMEMBER : we remove the first line, which is just meta
    #           so if we specify example  123  we  actually get line 124, and
    #           its zero indext thus 123 which is 124 becomes 125
    # TODO: Thus if given line number in text file subtract - 2 to get corresponding in generated
    # dataset array
    caseIndex = 125
    ########
    #
    # CLASSIFICATION
    #
    ########
    print("\nCLASSIFICATION STARTS HERE\n")
    sampleToClassify = create_data_set()[caseIndex - 2]
    sampleClass = sampleToClassify[-1]
    prediction = k_nearest_neighbor(sampleToClassify, k=10)

    print("The sample: " + str(sampleToClassify[:-1]) + " is predicted class: " + str(prediction))
    print(
        "The actual class is: " + str(sampleClass) + " and predicted is: " + str(prediction) +
        "\nThus the prediction is " + str(sampleClass == prediction)
    )

    ########
    #
    # REGRESSION
    #
    ########
    print("\nREGRESSION STARTS HERE\n")
    sampleToClassify = create_data_set(filename="regression")[caseIndex - 2]
    sampleClass = sampleToClassify[-1]
    regression_prediction = k_nearest_neighbor(sampleToClassify, k=10, regression=True, dataset="regression")

    print("The sample: " + str(sampleToClassify[:-1]) + " is predicted class: " + str(regression_prediction))
