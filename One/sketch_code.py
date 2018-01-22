import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from time import sleep
import numpy as np


# # Read in data
# directory =  "datasets/regression/"
# train_filename = directory + "train_2d_reg_data.csv"
# test_filename = directory + "test_2d_reg_data.csv"
#
#
# # dataset line : x1 x2 y (x, y, z)
# x1axis = []
# x2axis = []
# yaxis =[]
#
# with open(os.path.abspath(train_filename)) as f:
#     for l in f:
#         coord = l.split(",")
#         x1axis.append(float(coord[0]))
#         x2axis.append(float(coord[1]))
#         yaxis.append(float(coord[2]))
#
# # plot dataset
# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# ax = Axes3D(fig)
#
# ax.scatter(x1axis,x2axis,yaxis)
# ax.set_xlabel('X1 Label')
# ax.set_ylabel('X2 Label')
# ax.set_zlabel('Y Label')




####
# Algoritm
####
class Linear_regression():
    def __init__(self, trainingSet="train_2d_reg_data.csv", testingSet="test_2d_reg_data.csv"):
        self.directory =  "datasets/regression/"
        self.train_filename = self.directory + trainingSet
        self.test_filename = self.directory + testingSet

        self.trainingData, self.trainingClass = self.readData(self.train_filename)
        self.testingData, self.testingClass = self.readData(self.test_filename)
        self.weights = self.initWeights(self.trainingData, self.trainingClass)

        #
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def readData(self, dataset):
        with open(os.path.abspath(dataset)) as d:
            rows = ""
            classes = ""
            for l in d:
                l = l.split(",") # [x1...xn,y]
                row = "1"
                for feature in range(len(l)-1): #  [x1,x2,y] = 3 i want 0,1 == 3 - 1 = 2 => 0, 1
                    row += "," + str(l[feature])
                # matrix new row
                row += ";"
                rows += row
                classes += str(l[-1]) + ";" # add a new row with the class for this example
            # remove last ; from classes and rows
            rows = rows[:-1]
            classes = classes[:-1]

        return np.matrix(rows), np.matrix(classes)


    def plotData3d(self):
        self.ax.set_xlabel('X1 Label')
        self.ax.set_ylabel('X2 Label')
        self.ax.set_zlabel('Y Label')

        self.x1axis = []
        self.x2axis = []
        self.yaxis =[]

        example = 0
        for r in self.trainingData.getA():
            self.x1axis.append(float(r[1]))
            self.x2axis.append(float(r[2]))
            self.yaxis.append(float(self.trainingClass[example]))
            example += 1


        self.ax.scatter(self.x1axis,self.x2axis,self.yaxis)
        # self.fig.show()

    def plotLine3d(self):
        # hvis vi skal plott en linje, brukervi x1 og x2 fra alle eksempler, hvor y = vår h(x) for det treningseksemplet.

        calculatedY = []

        for r in self.trainingData:
            calculatedY.append(float(self.hypothesis(r)))

        # self.ax.scatter(self.x1axis, self.x2axis, calculatedY, color="red")
        self.ax.plot_trisurf(self.x1axis, self.x2axis, calculatedY, color="red")
        self.fig.show()

        print("Plottet surface")


    def info(self, e, cost):
        print("Epoc: ", e, "\nError: ", cost, "\nWeights: ", self.weights)
        print("\n"*2)


    def initWeights(self, X, y):
        return np.linalg.pinv(X.transpose() * X) * (X.transpose() * y)


    def hypothesis(self, xi):
        return self.weights.transpose() * xi.transpose()


    def cost_function(self, dataset, dataclass):
        s = 0
        numExamples = dataset.shape[0] # number of rows / traning examples
        a = 1 / numExamples
        for i in range(numExamples):
            #sum =  [ (h(x) - y)^2  ]
            res = (self.hypothesis(dataset[i]) - dataclass[i]) ** 2
            s += res.A[0][0]

        return a * s


    def train(self):
        print("Traning:\n")
        e = 0
        converged = False
        while e < 10 or converged:
            tempUpdates = [x for x in self.weights]
            cost = self.cost_function(self.trainingData, self.trainingClass)
            if cost < 0.1:
                converged = True
                self.info(e, cost)
                break

            # change the weight
            for temp in range(len(tempUpdates)):
                tempUpdates[temp] -= cost

            # updated simultaneously
            for w in range(len(self.weights)):
                self.weights[w] = tempUpdates[w]

            self.info(e, cost)
            e += 1
        print("DONE Training")



    def test(self):
        print("\n","Testing:\n")
        testingRes = self.cost_function(self.testingData, self.testingClass)
        print("Weights:\n" , self.weights, "\n", "Error: ", testingRes)



def main(task=1):
    lr = Linear_regression()
    if task == 1:
        lr.plotData3d()
    ######
    lr.train()
    lr.test()
    ######
    if task == 1:
        lr.plotLine3d()


    sleep(5)



main()
