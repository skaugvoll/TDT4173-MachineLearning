#from sklearn.datasets import load_digits
import math
import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layers, X, y, learning_rate):
        self.layer_count = len(hidden_layers)
        self.layers = hidden_layers
        self.X = X
        self.y = y
        self.weights = self.initialize_weights(X.shape[1])
        self.learning_rate = learning_rate

    def initialize_weights(self,weight_count):
        weights = []
        for i in range(weight_count):
            weights.append(0.5)
        return np.array(weights)

    def sigmoid(self,z):
        return 1 / (1 + math.pow(math.e,-z))

    def L(self,desired_output, actual_output):
        return 0.5 * math.pow(desired_output - actual_output,2)

#digits = load_digits()
X = np.mat([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

n = NeuralNetwork([2],X,y,0.1)

#print(np.dot(X[1],n.weights))

