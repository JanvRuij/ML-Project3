import numpy as np
import gurobipy as gp
from gurobipy import GRB


class QuadraticKnapSack:
    def __init__(self, n, c):
        # value of each item
        self.v = np.random.rand(1, n)
        # weight of each item
        self.w = np.random.rand(1, n)
        # combinatory value of item i and j
        self.V = np.random.rand(n, n)
        # capacatiy of the knapsack
        self.c = c
        # number or items in the knapsack
        self.n = n
        # solution array
        self.x = np.zeros(n)

    def ILP_Solver(self):
        model = gp.Model("QuadraticKnapSack")
        model.setParam("OutputFlag", 0)
        model.setParam("timelimit", 30)

        # create 1xn solution vector
        x = model.addvars(1, self.n, vtype=GRB.BINARY, name="x")



