from keras import metrics
from keras.src.engine.training import optimizer, optimizers
from keras.src.layers.attention.multi_head_attention import activation
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import BinaryCrossentropy


class QuadraticKnapSack:
    def __init__(self, n, c):
        # value of each item
        self.v = np.random.rand(n)
        # weight of each item
        self.w = np.random.rand(n)
        # combinatory value of item i and j
        self.V = np.random.rand(n, n)
        self.V = np.triu(self.V)
        # capacatiy of the knapsack
        self.c = c
        # number or items in the knapsack
        self.n = n

    def CLIN_Model(self):
        # create the model
        model = gp.Model("QuadraticKnapSack")
        model.setParam("OutputFlag", 0)
        model.setParam("timelimit", 30)

        # create 1xn solution vector
        x = model.addVars(self.n, vtype=GRB.BINARY, name="x")
        print(self.n)
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

        return model

    def ILP_Solver(self):
        # create the model
        model = self.CLIN_Model()
        # optimize the model
        model.optimize()
        # solution vector
        self.x = np.zeros(self.n)
        # read the output
        index = 0
        for var in model.getVars():
            if "x" in var.VarName:
                self.x[index] = var.X
                index += 1
        print(self.x)

        print(model.ObjVal)

    def MathHeuristic(self, threshold):
        # using NN to reduce the size of variables
        nn_model = keras.models.load_model("model_QK")
        predict = np.zeros(self.n)
        for i in range(self.n):
            high_combos = np.concatenate((self.V[:, i], self.V[i, :]))
            high_combos = np.sort(high_combos)[-100:]
            z = np.array([self.v[i], self.w[i]])
            X = np.append(z, high_combos)
            p = nn_model.predict(X[:, :5].reshape(1, 5), verbose='0')
            if p > threshold:
                predict[i] = 1

        # remove the variables
        self.n = np.count_nonzero(predict)
        self.v = self.v[np.dot(self.v, predict) != 0][0]
        self.w = self.w[np.dot(self.w, predict) != 0][0]
        non_zero = np.where(predict != 0)
        self.V = self.V[non_zero, :]
        self.V = self.V[:, non_zero][0][0]
        # now lets optimize the model with less variables
        gp_model = self.CLIN_Model()
        gp_model.optimize()
        print(gp_model.ObjVal)


# x = QuadraticKnapSack(300, 15)
# x.ILP_Solver()
# x.MathHeuristic(0.25)
# create the data
#with open("QKInstances.txt", "a") as file:
#    # create training data
#    for i in range(100):
#        print(i)
#        x = QuadraticKnapSack(200, 15)
#        x.ILP_Solver()
#        for j in range(200):
#            # write the value, weight and top 30 combinatory values of each item (X)
#            high_combos = np.concatenate((x.V[:, j], x.V[j, :]))
#            high_combos = np.sort(high_combos)[-100:]
#            # and write the outcome of the solver (Y)
#            file.write(f"[{x.v[j]},{x.w[j]},{','.join(map(str, high_combos))},{x.x[j]}]" + "\n")


with open("QKInstances.txt", "r") as file:
    lines = file.readlines()
    X = np.array([[float(num)
                  for num in string[1:-2]
                   .replace('-0.0', '0').split(",")[:-1]]
                  for string in lines])

    Y = np.array([[float(num)
                  for num in string[-5:-2]
                   .replace('-0.0', '0').split()]
                  for string in lines])

# number of data points
print(X.shape)
# create NN model
model = Sequential()
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
opt = SGD(learning_rate=0.005)
model.compile(loss=BinaryCrossentropy(),
              optimizer=opt, metrics=["accuracy"])

# train the model
model.fit(X[:15000, :7], Y[:15000], epochs=100, batch_size=300)

model.save("model_QK")
# predict with the keras model
for j in range(5, 95, 5):
    TP = 1
    TN = 1
    FP = 1
    FN = 1
    threshold = j/100
    prediction = model.predict(X[:, :7])
    for i in range(len(X) - 15000):
        if prediction[i] > threshold and Y[i] == 1:
            TP += 1
        elif prediction[i] > threshold and Y[i] == 0:
            FP += 1
        elif prediction[i] < threshold and Y[i] == 1:
            FN += 1
        elif prediction[i] < threshold and Y[i] == 0:
            TN += 1

    print(threshold, F"TP: {TP}",  F"FP: {FP}", F"TN: {TN}", F"FN: {FN}",
          round(100*(TP+TN)/(len(X)-15000), 2),
          round(100*FN/(FN+TN), 2))
