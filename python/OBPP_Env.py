# Create the knapsack environment
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# Parameters
N = 50
R = 100


class BalanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __init__(self):
        self.seed()
        self.state = self.reset()
        self.action_space = spaces.Discrete(N)
        self.observation_space = spaces.Box(
            low=np.array([0.0] * (4 + 2*N)),
            high=np.array([N, N*R, N*R, N*(R+10)] + [R, R+10] * N),
            dtype=np.uint32)

    def reset(self):
        self.items = []
        self.C = 0
        self.nbItems = int(random.randint(25, 50))
        for i in range(self.nbItems):
            self.items.append(random.randint(1, R))
            self.C += self.items[-1]
            self.items.append(self.items[-1] + 10)
        self.C = int(self.C * (0.1 + random.random() * 0.8))
        for i in range(self.nbItems, N):
            self.items.append(0)
            self.items.append(0)

        self.total_reward = 0
        self.total_weight = 0
        self.state = self._update_state()
        self.greedy = greedy(self.items, self.C)
        return self.state

    def step(self, action):
        reward = 0
        done = False
        self.total_weight += self.items[2*action]
        if self.items[2*action] == 0:
            reward = 0
            done = True
        elif self.total_weight > self.C:
            reward = 0
            done = True
        else:
            reward = self.items[2*action + 1]
            self.nbItems -= 1
            self.items[2*action] = 0
            self.items[2*action + 1] = 0
            self.items[2*action] = self.items[2*self.nbItems],
            self.items[2*self.nbItems] = self.items[2*action]
            self.items[2*action + 1] = self.items[2*self.nbItems + 1]
            self.items[2*self.nbItems + 1] = self.items[2*action + 1]

        self.total_reward += reward
        self.state = self._update_state()

        return self.state, reward, done, {}

    def _update_state(self):
        tempS = []
        tempS.append(self.nbItems)
        sumW = sum(self.items[2*i] for i in range(N))
        sumP = sum(self.items[2*i+1] for i in range(N))
        tempS.append(self.C - self.total_weight)
        tempS.append(sumW)
        tempS.append(sumP)
        state = np.array(tempS + self.items)
        return state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(
                round(self.total_reward, 3), "(",
                round(self.total_reward - self.greedy, 3), ")",
                round(self.total_weight/self.C, 3)
            )
        """if self.total_reward - self.greedy > 0:
            print(self.initialItem)
            print(self.solution)
            print(self.greedySol)"""
