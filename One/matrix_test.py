import numpy as np


X = np.matrix('1,2,4,5; 5,6,5,5') # 2 training examples
y = np.matrix('9;10') # target variable for 2 traning examples

Xtransposed = X.transpose() # transpose X


print("X: ", X)
print()
print("Xt: ", Xtransposed)
print()
print("y: ", y)
print()

# a = np.linalg.inv(Xtransposed * X)
# a = np.linalg.pinv(Xtransposed * X)
# print("a: ",a.shape,"\n", a)

# print()

# aTransposed = a.transpose()
# print("aTransposed: ", aTransposed.shape,"\n", aTransposed)


# print(Xtransposed, y)
# b =  Xtransposed * y
# print(b)
# w = a * b

# print("a: ", a)
# print("b: ", b)
# print("w: ", w)

####
# This works, maddafakka
####


def find_w():
    w = np.linalg.pinv(X.transpose() * X) * (X.transpose() * y)
    return w

def mean_squared_error(weights, X):
    s = 0
    for i in range(X.shape[0]): # x.shape = (R,C)
        pass

def hypothesis(w, x):
    return w.transpose() * x
