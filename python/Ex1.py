import numpy as np
import gurobipy as gp
from gurobipy import GRB


class QuadraticKnapSack:
    def __init__(self, n, c):
        # value of each item
        self.v = np.random.rand(n)
        # weight of each item
        self.w = np.random.rand(n)
        # combinatory value of item i and j
        self.V = np.random.rand(n, n)
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

        # set the objective
        model.setObjective(
                gp.quicksum(self.v[i] * x[i] for i in range(self.n))
                +
                gp.quicksum(self.V[i][j] * x[i] * x[j]
                            for i in range(self.n) for j in range(i, self.n)),
                sense=GRB.MAXIMIZE)

        # add capacity constraint
        model.addConstr(
                gp.quicksum(self.w[i] * x[i] for i in range(0, self.n))
                <= self.c)

        # optimize the model
        model.optimize()


x = QuadraticKnapSack(56, 10)
x.ILP_Solver()
