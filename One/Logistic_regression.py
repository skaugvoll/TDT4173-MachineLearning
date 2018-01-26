import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import os
from time import sleep
import numpy as np
import math



####
# Algoritm
####
'''
This need some refactoring!


'''
class Logistic_regression():
    def __init__(self, trainingSet="cl_train_1.csv", testingSet="cl_test_1.csv", learningRate=0.1):
        self.directory =  "datasets/classification/"
        self.train_filename = self.directory + trainingSet
        self.test_filename = self.directory + testingSet

        self.trainingData, self.trainingClass = self.readData(self.train_filename)
        self.testingData, self.testingClass = self.readData(self.test_filename)

        self.weights = self.initWeights(self.trainingData, self.trainingClass)
        self.learningRate = learningRate

        self.errors = []
        self.testError = []

        # print(self.testingClass)

        #


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



    def plotData2d(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xlabel('X Label')

        self.xPosaxis = []
        self.yPosaxis =[]

        self.xNegaxis = []
        self.yNegaxis =[]

        example = 0
        for r in self.trainingData.getA():
            # print(r)
            if(self.trainingClass[example] == 0):
                self.xPosaxis.append(float(r[1]))
                self.yPosaxis.append(float(r[2]))
            else:
                self.xNegaxis.append(float(r[1]))
                self.yNegaxis.append(float(r[2]))

            # self.x1axis.append(float(r[1]))
            # self.yaxis.append(float(r[2]))
            example += 1

        # self.ax.scatter(self.x1axis,self.yaxis)
        self.ax.scatter(self.xPosaxis,self.yPosaxis)
        self.ax.scatter(self.xNegaxis,self.yNegaxis)
        # self.fig.show()


    def plotLine2d(self):
        x = np.linspace(0,1)
        y = (self.weights[0] + self.weights[1]*x) / (-self.weights[2])

        self.ax.plot(x, y.getA1())
        # self.fig.show()

        print("Plottet line 2d")

    def plotError(self, errors):
        fig = plt.figure()
        plt.plot([ x for x in range(1, len(errors)+1)], errors)
        # plt.show()


    def info(self, epoc, cost):
        print("Epoc: ", epoc, "\nError: ", cost, "\nWeights: ", self.weights)
        print("\n"*2)


    def initWeights(self, X, y):
        return np.matrix("0;0;0")

    def hypothesis(self, xi):
        res = self.weights.transpose() * xi.transpose() # transpose xi from [  ] to []

        return res

    def sigmoid(self, hypothesisRes):
        res = 1 / (1 + np.exp(-hypothesisRes))
        return res.getA1()[0]

    def probabilityDistribution(self, y, xi):
        y = y.getA1()[0]
        a = self.sigmoid(self.hypothesis(xi)).getA1()[0] # ==> o-(wTx)

        return (a ** y) * (1 - a) ** (1 - y)

    def likelihood(self, dataset, classes):
        l = 0
        for i in range(len(dataset)):
            hypres = self.hypothesis(dataset[i])
            z = self.sigmoid(hypres)
            output = classes.getA1()[i]
            l +=  output * math.log(z) + (1 - output * math.log(1 - z))
        return l

    def crossEntropy(self, dataset, classes):
        l = 0
        for i in range(len(dataset)):
            hypres = self.hypothesis(dataset[i])
            z = self.sigmoid(hypres)
            # print("z: ", z)
            output = classes.getA1()
            l += output[i] * math.log(z) + (1 - output[i]) * math.log(1 - z)

        return -(l  / len(dataset))


        # res = -(self.likelihood(dataset, classes) / len(dataset))

        # print(res)
        # return res


    def updateRule(self):
        summation = 0
        numExamples = self.trainingData.shape[0] # number of rows / traning examples
        for i in range(numExamples):
            # print("step: ", i)
            res = (self.sigmoid(self.hypothesis(self.trainingData[i])) - self.trainingClass[i]) * self.trainingData[i]
            # print("RES updRl: ", res)
            summation += res

        # print("HERE COMES W and S")
        # print("W1: ",self.weights)
        # print()
        # print("S: ",summation * self.learningRate)

        self.weights = self.weights.transpose() - (self.learningRate * summation)
        self.weights = self.weights.transpose()
        # print("W2: ",self.weights)


    def train(self):
        print("Traning:\n")
        e = 0
        maxEpocs = 1000
        converged = False


        while e < maxEpocs:
            # change the weight
            self.updateRule() # update the weights

            # remeber error
            self.errors.append(self.crossEntropy(self.trainingData, self.trainingClass))
            # self.info(e, cost)
            e += 1

        self.info(e,self.errors[-1])
        print("DONE Training")
        self.plotError(self.errors)



    def test(self):
        print("\n","Testing:\n")

        testRes = self.crossEntropy(self.testingData, self.testingClass)


        print("Weights:\n" , self.weights, "\n", "Error: ", testRes)
        print("DONE Testing")


    def plot(self):
        plt.show()



def main(task=1):
    if task == 2:
        lr = Logistic_regression(trainingSet="cl_train_2.csv", testingSet="cl_test_2.csv") # 3d

    else:
        lr = Logistic_regression()

    lr.plotData2d()
    ######
    lr.train()
    lr.test()
    ######

    lr.plotLine2d()
    lr.plot()
    sleep(10)



if __name__ == "__main__":
    main(2) # 0 = 2d, 1 = 3d
