import numpy as np
import gurobipy as gp
from gurobipy import GRB
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense

class QuadraticKnapSack:
    def __init__(self, n, c):
        # value of each item
        self.v = np.random.randint(1, 50, size=n)
        # weight of each item
        self.w = np.random.randint(1, 50, size=n)
        # combinatory value of item i and j
        self.V = np.random.randint(1, 50, size=(n, n))
        # capacatiy of the knapsack
        self.c = c
        # number or items in the knapsack
        self.n = n
        # solution array
        self.x = np.zeros(n)

    def ILP_Solver(self):
        # create the model
        model = gp.Model("QuadraticKnapSack")
        # model.setParam("OutputFlag", 0)
        # model.setParam("timelimit", 30)

        # create 1xn solution vector
        x = model.addVars(self.n, vtype=GRB.BINARY, name="x")
        w = model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="w")

        # set the objective
        model.setObjective(
                gp.quicksum(self.v[i] * x[i] for i in range(self.n))
                +
                gp.quicksum(self.V[i][j] * w[i, j]
                            for i in range(self.n - 1)
                            for j in range(i + 1, self.n)),
                sense=GRB.MAXIMIZE)

        # add capacity constraint
        model.addConstr(
                gp.quicksum(self.w[i] * x[i] for i in range(0, self.n))
                <= self.c)
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                model.addConstr(w[i, j] <= x[i])
                model.addConstr(w[i, j] <= x[j])

        # optimize the model
        model.optimize()


# create NN model
model = Sequential()
model.add(Dense(9, input_dim=14, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X,Y, epochs=50, batch_size=256) 
