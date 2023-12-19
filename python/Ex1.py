import numpy as np
import gurobipy as gp
from gurobipy import GRB


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


x = QuadraticKnapSack(230, 1000)
x.ILP_Solver()
